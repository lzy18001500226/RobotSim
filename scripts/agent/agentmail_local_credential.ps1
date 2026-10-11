param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Set", "Emit", "SelfTestEmit", "Remove")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
# WSL-launched Windows PowerShell can inherit Codex runtime modules that shadow its built-ins.
$env:PSModulePath = Join-Path $PSHOME "Modules"
Import-Module Microsoft.PowerShell.Security -ErrorAction Stop
$credentialRoot = Join-Path $env:LOCALAPPDATA "RobotSim\AgentMail"
$keyPath = Join-Path $credentialRoot "api-key.dpapi"
$settingsPath = Join-Path $credentialRoot "settings.json"

function Set-PrivateAcl([string]$Path, [bool]$IsDirectory) {
    $acl = Get-Acl -LiteralPath $Path
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($existingRule in @($acl.Access)) {
        [void]$acl.RemoveAccessRuleSpecific($existingRule)
    }
    $sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    $acl.SetOwner($sid)
    if ($IsDirectory) {
        $inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
            [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
    } else {
        $inheritance = [System.Security.AccessControl.InheritanceFlags]::None
    }
    $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
        $sid,
        [System.Security.AccessControl.FileSystemRights]::FullControl,
        $inheritance,
        [System.Security.AccessControl.PropagationFlags]::None,
        [System.Security.AccessControl.AccessControlType]::Allow
    )
    [void]$acl.AddAccessRule($rule)
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Ensure-CredentialRoot {
    if (-not (Test-Path -LiteralPath $credentialRoot -PathType Container)) {
        [void][System.IO.Directory]::CreateDirectory($credentialRoot)
    }
    Set-PrivateAcl $credentialRoot $true
}

function Convert-SecureStringToPlainText([System.Security.SecureString]$Value) {
    $pointer = [IntPtr]::Zero
    try {
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    } finally {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
    }
}

function Write-AtomicText([string]$Path, [string]$Value, [System.Text.Encoding]$Encoding) {
    $temporaryPath = "$Path.$([guid]::NewGuid().ToString('N')).tmp"
    try {
        [System.IO.File]::WriteAllText($temporaryPath, $Value, $Encoding)
        Move-Item -LiteralPath $temporaryPath -Destination $Path -Force
        Set-PrivateAcl $Path $false
    } finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

function Get-Settings {
    if (-not (Test-Path -LiteralPath $settingsPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $keyPath -PathType Leaf)) {
        throw "Local AgentMail configuration is not present."
    }
    $settings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($settings.version -ne 1 -or
        [string]::IsNullOrWhiteSpace([string]$settings.inbox_id) -or
        [string]::IsNullOrWhiteSpace([string]$settings.notify_to)) {
        throw "Local AgentMail configuration is invalid."
    }
    return $settings
}

function Emit-Bundle([string]$Key, [string]$InboxId, [string]$Recipient) {
    $bundle = [ordered]@{
        AGENTMAIL_API_KEY = $Key
        AGENTMAIL_INBOX_ID = $InboxId
        ROBOTSIM_NOTIFY_TO = $Recipient
    } | ConvertTo-Json -Compress
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bundle))
    [Console]::Out.WriteLine($encoded)
}

try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw "Windows user credential protection is required."
    }

    switch ($Action) {
        "Set" {
            Ensure-CredentialRoot
            $secureKey = Read-Host "AgentMail API key (input is hidden)" -AsSecureString
            if ($secureKey.Length -eq 0) {
                throw "The AgentMail API key cannot be empty."
            }
            $inboxId = Read-Host "AgentMail inbox ID"
            $recipient = Read-Host "Notification recipient email"
            if ([string]::IsNullOrWhiteSpace($inboxId) -or
                $inboxId -match "\s" -or
                $recipient -notmatch "^[^@\s]+@[^@\s]+$") {
                throw "The inbox ID or recipient format is invalid."
            }

            $encryptedKey = ConvertFrom-SecureString -SecureString $secureKey
            $settings = [ordered]@{
                version = 1
                inbox_id = $inboxId
                notify_to = $recipient
            } | ConvertTo-Json -Compress
            Write-AtomicText $keyPath $encryptedKey ([Text.Encoding]::ASCII)
            Write-AtomicText $settingsPath $settings (New-Object System.Text.UTF8Encoding($false))
            Write-Output "AgentMail local credential stored with current-user DPAPI protection."
        }
        "Emit" {
            $settings = Get-Settings
            $encryptedKey = Get-Content -LiteralPath $keyPath -Raw -Encoding ASCII
            $secureKey = ConvertTo-SecureString -String $encryptedKey
            $plainKey = Convert-SecureStringToPlainText $secureKey
            Emit-Bundle $plainKey ([string]$settings.inbox_id) ([string]$settings.notify_to)
            $plainKey = $null
        }
        "SelfTestEmit" {
            $testKey = "synthetic-agentmail-" + [guid]::NewGuid().ToString("N")
            $secureKey = ConvertTo-SecureString -String $testKey -AsPlainText -Force
            $encryptedKey = ConvertFrom-SecureString -SecureString $secureKey
            $roundTrip = ConvertTo-SecureString -String $encryptedKey
            $roundTripText = Convert-SecureStringToPlainText $roundTrip
            if ($roundTripText -cne $testKey) {
                throw "Synthetic DPAPI round-trip failed."
            }
            Emit-Bundle $roundTripText "synthetic-inbox@example.invalid" "synthetic-recipient@example.invalid"
            $testKey = $null
            $roundTripText = $null
        }
        "Remove" {
            foreach ($path in @($keyPath, $settingsPath)) {
                if (Test-Path -LiteralPath $path -PathType Leaf) {
                    Remove-Item -LiteralPath $path -Force
                }
            }
            if ((Test-Path -LiteralPath $credentialRoot -PathType Container) -and
                -not (Get-ChildItem -LiteralPath $credentialRoot -Force | Select-Object -First 1)) {
                Remove-Item -LiteralPath $credentialRoot -Force
            }
            Write-Output "RobotSim AgentMail local credential files removed."
        }
    }
    exit 0
} catch {
    [Console]::Error.WriteLine("AgentMail local credential operation failed.")
    exit 1
}
