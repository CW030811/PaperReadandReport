param(
    [string]$Remote = 'origin',
    [string]$Branch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:RepoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-Git {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    & git -C $script:RepoRoot @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-CurrentBranch {
    $name = (& git -C $script:RepoRoot rev-parse --abbrev-ref HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to detect the current branch.'
    }
    if ($name -eq 'HEAD') {
        throw 'Detached HEAD is not supported. Check out a branch first.'
    }
    return $name
}

function Ensure-RemoteExists {
    param([string]$Name)

    $remoteUrl = (& git -C $script:RepoRoot config --get ("remote.{0}.url" -f $Name) 2> $null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($remoteUrl -join '').Trim())) {
        throw "Git remote '$Name' is not configured. Add it first, for example: git remote add $Name <repo-url>"
    }
}

Ensure-RemoteExists -Name $Remote

if (-not $Branch) {
    $Branch = Get-CurrentBranch
}

& git -C $script:RepoRoot diff --quiet
if ($LASTEXITCODE -eq 1) {
    throw 'Working tree has local changes. Commit or stash them before pulling.'
}
elseif ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect working tree changes.'
}

Invoke-Git pull --rebase $Remote $Branch

Write-Output "Updated '$Branch' from '$Remote'."
