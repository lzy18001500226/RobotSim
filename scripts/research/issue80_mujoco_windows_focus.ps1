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
    [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr parent, EnumWindowsProc callback, IntPtr lParam);
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

    public static List<Dictionary<string, object>> VisibleChildWindows(IntPtr parent) {
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
                {"hwnd", hwnd.ToInt64()}, {"parent_hwnd", parent.ToInt64()}, {"pid", pid},
                {"title", title.ToString()}, {"class_name", cls.ToString()}, {"visible", true},
                {"width", rect.Right - rect.Left}, {"height", rect.Bottom - rect.Top},
                {"process_name", processName}, {"process_path", processPath}
            });
            return true;
        };
        EnumChildWindows(parent, callback, IntPtr.Zero);
        GC.KeepAlive(callback);
        return items;
    }
}
"@

Add-Type -TypeDefinition $native -ErrorAction Stop
$topLevelWindows = [RobotSimX2WindowsFocus]::VisibleWindows()
$allVisible = @($topLevelWindows)
foreach ($parentWindow in $topLevelWindows) {
    $children = [RobotSimX2WindowsFocus]::VisibleChildWindows([IntPtr]([long]$parentWindow.hwnd))
    foreach ($childWindow in $children) {
        $childWindow["parent_title"] = $parentWindow.title
        $childWindow["parent_process_name"] = $parentWindow.process_name
        $allVisible += $childWindow
    }
}
$expectedAspect = if ($ExpectedHeight -gt 0) { [double]$ExpectedWidth / $ExpectedHeight } else { 0.0 }
$windowMatches = @($allVisible | Where-Object {
    $_.title -ceq $ExpectedTitle -and
    $_.width -gt 200 -and $_.height -gt 200 -and
    $expectedAspect -gt 0 -and
    [Math]::Abs(([double]$_.width / $_.height) - $expectedAspect) -lt 0.08
})
$foregroundBefore = [RobotSimX2WindowsFocus]::GetForegroundWindow().ToInt64()
$foregroundInfo = @($allVisible | Where-Object { $_.hwnd -eq $foregroundBefore }) | Select-Object -First 1
$relatedWindows = @($allVisible | Where-Object {
    $_.title -match "(?i)MuJoCo|Remote Desktop" -or
    $_.process_name -match "(?i)^(msrdc|mstsc|wslg|wslhost)$"
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
    matching_visible_windows = $windowMatches
    related_wslg_windows = $relatedWindows
    foreground_hwnd_before = $foregroundBefore
    foreground_window_before = $foregroundInfo
    foreground_hwnd_after = $null
    set_foreground_window_return = $null
    status = "WINDOW_MATCH_NOT_UNIQUE"
    key_events_sent = 0
    activation_scope = "SetForegroundWindow only; no Windows keys or mouse events are synthesized"
}
if ($windowMatches.Count -eq 1) {
    $matchedWindow = $windowMatches[0]
    $target = [IntPtr]([long]$matchedWindow.hwnd)
    $activationTarget = if ($matchedWindow.parent_hwnd) {
        [IntPtr]([long]$matchedWindow.parent_hwnd)
    } else {
        $target
    }
    $result.activation_window_hwnd = $activationTarget.ToInt64()
    if ($Mode -eq "Activate") {
        $result.set_foreground_window_return = [RobotSimX2WindowsFocus]::SetForegroundWindow($activationTarget)
        Start-Sleep -Milliseconds 100
    }
    $foreground = [RobotSimX2WindowsFocus]::GetForegroundWindow().ToInt64()
    $result.foreground_hwnd_after = $foreground
    $stillPresent = $false
    foreach ($currentWindow in [RobotSimX2WindowsFocus]::VisibleWindows()) {
        if ($currentWindow.hwnd -eq $matchedWindow.hwnd -and $currentWindow.title -ceq $ExpectedTitle) {
            $stillPresent = $true
        }
        foreach ($currentChild in [RobotSimX2WindowsFocus]::VisibleChildWindows([IntPtr]([long]$currentWindow.hwnd))) {
            if ($currentChild.hwnd -eq $matchedWindow.hwnd -and $currentChild.title -ceq $ExpectedTitle) {
                $stillPresent = $true
            }
        }
    }
    if ($Mode -eq "Inspect") {
        $result.status = "UNIQUE_VISIBLE_WINDOW_INSPECTED"
    } elseif ($stillPresent -and $foreground -eq $activationTarget.ToInt64()) {
        $result.status = "FOREGROUND_VERIFIED"
    } else {
        $result.status = "FOREGROUND_NOT_VERIFIED"
    }
}
$json = $result | ConvertTo-Json -Depth 8 -Compress
$jsonAscii = [System.Text.RegularExpressions.Regex]::Replace(
    $json,
    "[^\x00-\x7F]",
    [System.Text.RegularExpressions.MatchEvaluator]{
        param($match)
        return "\u{0:X4}" -f [int][char]$match.Value[0]
    }
)
[Console]::Out.WriteLine($jsonAscii)
