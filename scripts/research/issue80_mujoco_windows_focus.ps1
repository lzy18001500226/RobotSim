param(
    [Parameter(Mandatory = $true)][string]$ExpectedTitle,
    [Parameter(Mandatory = $true)][int]$ExpectedX11Pid,
    [Parameter(Mandatory = $true)][long]$ExpectedX11WindowId,
    [Parameter(Mandatory = $true)][int]$ExpectedWidth,
    [Parameter(Mandatory = $true)][int]$ExpectedHeight,
    [ValidateSet("Inspect", "Activate")][string]$Mode = "Inspect"
)

$ErrorActionPreference = "Stop"
$native = @"
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

public static class RobotSimX2WindowsFocus {
    public delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr hwnd, StringBuilder text, int maxCount);
    [DllImport("user32.dll")] public static extern int GetWindowTextLength(IntPtr hwnd);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr hwnd, StringBuilder text, int maxCount);
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd, out RECT rect);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();

    public static List<Dictionary<string, object>> VisibleWindows() {
        var items = new List<Dictionary<string, object>>();
        EnumWindowsProc callback = (hwnd, lParam) => {
            if (!IsWindowVisible(hwnd)) return true;
            int length = GetWindowTextLength(hwnd);
            if (length <= 0) return true;
            var title = new StringBuilder(length + 1);
            var cls = new StringBuilder(256);
            GetWindowText(hwnd, title, title.Capacity);
            GetClassName(hwnd, cls, cls.Capacity);
            RECT rect;
            if (!GetWindowRect(hwnd, out rect)) return true;
            uint pid;
            GetWindowThreadProcessId(hwnd, out pid);
            string processName = "";
            string processPath = "";
            try {
                var process = Process.GetProcessById((int)pid);
                processName = process.ProcessName;
                try { processPath = process.MainModule.FileName; } catch { processPath = "UNAVAILABLE"; }
            } catch { processName = "UNAVAILABLE"; }
            items.Add(new Dictionary<string, object> {
                {"hwnd", hwnd.ToInt64()}, {"pid", pid}, {"title", title.ToString()},
                {"class_name", cls.ToString()}, {"visible", true},
                {"width", rect.Right - rect.Left}, {"height", rect.Bottom - rect.Top},
                {"process_name", processName}, {"process_path", processPath}
            });
            return true;
        };
        EnumWindows(callback, IntPtr.Zero);
        GC.KeepAlive(callback);
        return items;
    }
}
"@

Add-Type -TypeDefinition $native -ErrorAction Stop
$allVisible = [RobotSimX2WindowsFocus]::VisibleWindows()
$expectedAspect = if ($ExpectedHeight -gt 0) { [double]$ExpectedWidth / $ExpectedHeight } else { 0.0 }
$matches = @($allVisible | Where-Object {
    $_.title -ceq $ExpectedTitle -and
    $_.width -gt 200 -and $_.height -gt 200 -and
    $expectedAspect -gt 0 -and
    [Math]::Abs(([double]$_.width / $_.height) - $expectedAspect) -lt 0.08
})
$result = [ordered]@{
    schema = "robotsim.issue80.windows-foreground-focus.v1"
    mode = $Mode
    x11_target = [ordered]@{
        pid = $ExpectedX11Pid
        window_id = $ExpectedX11WindowId
        title = $ExpectedTitle
        width = $ExpectedWidth
        height = $ExpectedHeight
    }
    matching_visible_windows = $matches
    foreground_hwnd_before = [RobotSimX2WindowsFocus]::GetForegroundWindow().ToInt64()
    foreground_hwnd_after = $null
    set_foreground_window_return = $null
    status = "WINDOW_MATCH_NOT_UNIQUE"
    key_events_sent = 0
    activation_scope = "SetForegroundWindow only; no Windows keys or mouse events are synthesized"
}
if ($matches.Count -eq 1) {
    $target = [IntPtr]([long]$matches[0].hwnd)
    if ($Mode -eq "Activate") {
        $result.set_foreground_window_return = [RobotSimX2WindowsFocus]::SetForegroundWindow($target)
        Start-Sleep -Milliseconds 100
    }
    $foreground = [RobotSimX2WindowsFocus]::GetForegroundWindow().ToInt64()
    $result.foreground_hwnd_after = $foreground
    $stillUnique = @([RobotSimX2WindowsFocus]::VisibleWindows() | Where-Object {
        $_.hwnd -eq $matches[0].hwnd -and $_.title -ceq $ExpectedTitle
    })
    if ($Mode -eq "Inspect") {
        $result.status = "UNIQUE_VISIBLE_WINDOW_INSPECTED"
    } elseif ($stillUnique.Count -eq 1 -and $foreground -eq [long]$matches[0].hwnd) {
        $result.status = "FOREGROUND_VERIFIED"
    } else {
        $result.status = "FOREGROUND_NOT_VERIFIED"
    }
}
$result | ConvertTo-Json -Depth 8 -Compress
