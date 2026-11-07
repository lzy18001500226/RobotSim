param([string]$ExpectedTitle="",[switch]$ProbeOnly)
# Activate exactly one WSLg MuJoCo window. No keys, mouse or global settings.
$ErrorActionPreference='Stop'
if (-not $ProbeOnly -and $ExpectedTitle -notmatch '^MuJoCo : [A-Za-z0-9_. ()-]+$') { throw 'Unsupported MuJoCo title' }
Add-Type -TypeDefinition @'
using System;using System.Text;using System.Runtime.InteropServices;
public static class RobotSimVerifiedForeground {
 public delegate bool EnumFn(IntPtr w,IntPtr l);
 [DllImport("user32.dll")] public static extern bool EnumWindows(EnumFn f,IntPtr l);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr w,StringBuilder s,int n);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr w,out uint pid);
 [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr w);
 [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr w);
 [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr w,int cmd);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr w);
 [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
 [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int key);
}
'@
if($ProbeOnly){
 @{status='READ_ONLY_POINTER_PROBE';LeftDown=([RobotSimVerifiedForeground]::GetAsyncKeyState(1)-band 0x8000)-ne 0;RightDown=([RobotSimVerifiedForeground]::GetAsyncKeyState(2)-band 0x8000)-ne 0;MiddleDown=([RobotSimVerifiedForeground]::GetAsyncKeyState(4)-band 0x8000)-ne 0;keyboardEvents=0;mouseEvents=0;globalSettingsChanged=$false}|ConvertTo-Json -Compress
 exit 0
}
$wanted=$ExpectedTitle+' (Ubuntu-22.04)'
$found=[Collections.Generic.List[object]]::new()
[void][RobotSimVerifiedForeground]::EnumWindows({param($w,$unused)
 $sb=[Text.StringBuilder]::new(1024);[void][RobotSimVerifiedForeground]::GetWindowText($w,$sb,$sb.Capacity)
 if($sb.ToString() -ceq $wanted -and [RobotSimVerifiedForeground]::IsWindowVisible($w)){
  $owner=[uint32]0;[void][RobotSimVerifiedForeground]::GetWindowThreadProcessId($w,[ref]$owner)
  $p=Get-Process -Id $owner -ErrorAction Stop
  if($p.Path -ieq 'C:\Program Files\WSL\msrdc.exe'){$found.Add([pscustomobject]@{Handle=$w;Owner=$owner;Path=$p.Path;Title=$sb.ToString()})}
 }
 return $true
},[IntPtr]::Zero)
if($found.Count -ne 1){throw ('Expected one exact title/path-verified WSLg window, got '+$found.Count)}
$target=$found[0];$ownerNow=[uint32]0
[void][RobotSimVerifiedForeground]::GetWindowThreadProcessId($target.Handle,[ref]$ownerNow)
if($ownerNow -ne $target.Owner){throw 'Window owner changed'}
if([RobotSimVerifiedForeground]::IsIconic($target.Handle)){[void][RobotSimVerifiedForeground]::ShowWindowAsync($target.Handle,9)}
$sent=[RobotSimVerifiedForeground]::SetForegroundWindow($target.Handle)
$timer=[Diagnostics.Stopwatch]::StartNew()
do {
 $actual=[RobotSimVerifiedForeground]::GetForegroundWindow()
 if($actual -eq $target.Handle){break}
 Start-Sleep -Milliseconds 20
}while($timer.Elapsed.TotalSeconds -lt .75)
$ok=$actual -eq $target.Handle
@{status=$(if($ok){'FOREGROUND_VERIFIED'}else{'FOREGROUND_NOT_CONFIRMED'});title=$target.Title;ownerPid=$target.Owner;ownerPath=$target.Path;targetHandle=$target.Handle.ToInt64();actualForeground=$actual.ToInt64();setForegroundReturn=$sent;keyboardEvents=0;mouseEvents=0;globalSettingsChanged=$false}|ConvertTo-Json -Compress
if(-not $ok){exit 2}
