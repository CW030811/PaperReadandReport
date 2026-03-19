param(
    [Parameter(Mandatory = $true)]
    [string]$PaperDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$templateDir = Join-Path $repoRoot 'templates'

$requiredFiles = @(
    '07_guided_reading_cn/reading_map.md',
    '07_guided_reading_cn/guided_reading_rules.md',
    '07_guided_reading_cn/progress_log.md'
)

$readingMapHeadings = @(
    '## Paper',
    '## Reading Goal',
    '## Chapter Plan',
    '## Unlock Rule',
    '## Notes for Codex'
)

$rulesHeadings = @(
    '## Purpose',
    '## Generation Policy',
    '## Revision Policy',
    '## Graduation Rule',
    '## Chapter Writing Contract'
)

$progressHeadings = @(
    '## Status Snapshot',
    '## Chapter History',
    '## Recurring Questions',
    '## Mastery Signals'
)

$chapterHeadings = @(
    [regex]::Matches(
        (Get-Content -Raw -Encoding UTF8 (Join-Path $templateDir 'guided_reading_chapter.md')),
        '(?m)^## .+$'
    ) | ForEach-Object { $_.Value }
)

$paperPath = Resolve-Path -LiteralPath $PaperDir -ErrorAction Stop | Select-Object -ExpandProperty Path
$errors = New-Object System.Collections.Generic.List[string]

function Add-CheckError {
    param([string]$Message)
    [void]$script:errors.Add($Message)
}

function Require-File {
    param([string]$RelativePath)
    $target = Join-Path $paperPath ($RelativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
        Add-CheckError "Missing guided-reading file: $RelativePath"
    }
    return $target
}

function Require-Headings {
    param(
        [string]$Path,
        [string[]]$Headings
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    $content = Get-Content -Raw -Encoding UTF8 $Path
    foreach ($heading in $Headings) {
        if (-not $content.Contains($heading)) {
            Add-CheckError "Missing heading in $([System.IO.Path]::GetFileName($Path)): $heading"
        }
    }
}

function Check-ActiveStatus {
    param([string]$ReadingMapPath)
    if (-not (Test-Path -LiteralPath $ReadingMapPath -PathType Leaf)) {
        return
    }
    $activeLike = 0
    foreach ($line in (Get-Content -Encoding UTF8 $ReadingMapPath)) {
        if (-not $line.StartsWith('| Chapter')) {
            continue
        }
        $cells = @($line.Trim('|').Split('|') | ForEach-Object { $_.Trim() })
        if ($cells.Count -lt 5) {
            continue
        }
        if ($cells[-1] -in @('active', 'reviewing')) {
            $activeLike += 1
        }
    }
    if ($activeLike -gt 1) {
        Add-CheckError 'reading_map.md has more than one active/reviewing chapter'
    }
}

function Check-Chapters {
    param([string]$GuidedDir)
    if (-not (Test-Path -LiteralPath $GuidedDir -PathType Container)) {
        return
    }
    Get-ChildItem -LiteralPath $GuidedDir -Filter 'Chapter*.md' | Sort-Object Name | ForEach-Object {
        Require-Headings -Path $_.FullName -Headings $chapterHeadings
    }
}

$required = @{}
foreach ($relative in $requiredFiles) {
    $required[$relative] = Require-File -RelativePath $relative
}

Require-Headings -Path $required['07_guided_reading_cn/reading_map.md'] -Headings $readingMapHeadings
Require-Headings -Path $required['07_guided_reading_cn/guided_reading_rules.md'] -Headings $rulesHeadings
Require-Headings -Path $required['07_guided_reading_cn/progress_log.md'] -Headings $progressHeadings
Check-ActiveStatus -ReadingMapPath $required['07_guided_reading_cn/reading_map.md']
Check-Chapters -GuidedDir (Join-Path $paperPath '07_guided_reading_cn')

if ($errors.Count -gt 0) {
    foreach ($errorMessage in $errors) {
        Write-Output "[FAIL] $errorMessage"
    }
    exit 1
}

Write-Output '[OK] Guided-reading workspace passes the checks.'
