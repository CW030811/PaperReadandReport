param(
    [Parameter(Mandatory = $true)]
    [string]$PaperDir,
    [string]$Title,
    [switch]$Force,
    [switch]$ChapterTemplate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$templateDir = Join-Path $repoRoot 'templates'
$paperPath = Resolve-Path -LiteralPath $PaperDir -ErrorAction Stop | Select-Object -ExpandProperty Path

$fileMap = [ordered]@{
    'reading_map.md' = 'guided_reading_map.md'
    'guided_reading_rules.md' = 'guided_reading_rules.md'
    'progress_log.md' = 'guided_reading_progress_log.md'
}

function Get-TemplateText {
    param([string]$TemplateName)
    return Get-Content -Raw -Encoding UTF8 (Join-Path $templateDir $TemplateName)
}

function Render-Template {
    param(
        [string]$Text,
        [string]$PaperTitle,
        [string]$PaperFolder,
        [string]$ScanDate
    )
    $rendered = $Text.Replace('{{PAPER_TITLE}}', $PaperTitle)
    $rendered = $rendered.Replace('{{PAPER_FOLDER}}', $PaperFolder)
    $rendered = $rendered.Replace('{{SCAN_DATE}}', $ScanDate)
    return $rendered
}

function Write-Utf8File {
    param(
        [string]$Path,
        [string]$Content
    )
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Get-InferredTitle {
    param(
        [string]$WorkspacePath,
        [string]$OverrideTitle
    )
    if ($OverrideTitle) {
        return $OverrideTitle
    }

    $schemaPath = Join-Path $WorkspacePath '03_extraction\paper_schema.json'
    if (Test-Path -LiteralPath $schemaPath -PathType Leaf) {
        try {
            $schema = Get-Content -Raw -Encoding UTF8 $schemaPath | ConvertFrom-Json
            if ($schema.paper_title -and ([string]$schema.paper_title).Trim()) {
                return ([string]$schema.paper_title).Trim()
            }
        }
        catch {
        }
    }

    $briefPath = Join-Path $WorkspacePath '03_extraction\paper_brief.md'
    if (Test-Path -LiteralPath $briefPath -PathType Leaf) {
        foreach ($line in (Get-Content -Encoding UTF8 $briefPath)) {
            if ($line.StartsWith('- Title:')) {
                $candidate = $line.Split(':', 2)[1].Trim()
                if ($candidate) {
                    return $candidate
                }
            }
        }
    }

    return [System.IO.Path]::GetFileName($WorkspacePath)
}

function Write-IfNeeded {
    param(
        [string]$Target,
        [string]$Content,
        [bool]$ShouldForce
    )
    if ((Test-Path -LiteralPath $Target) -and -not $ShouldForce) {
        return $false
    }
    Write-Utf8File -Path $Target -Content $Content
    return $true
}

try {
    $paperFolder = (Resolve-Path -LiteralPath $paperPath).Path.Substring($repoRoot.Length + 1).Replace('\', '/')
}
catch {
    $paperFolder = $paperPath.Replace('\', '/')
}

$paperTitle = Get-InferredTitle -WorkspacePath $paperPath -OverrideTitle $Title
$scanDate = if ([System.IO.Path]::GetFileName($paperPath) -match '^([^_]+)_') { $Matches[1] } else { '' }
$guidedDir = Join-Path $paperPath '07_guided_reading_cn'
New-Item -ItemType Directory -Force -Path $guidedDir | Out-Null

$created = New-Object System.Collections.Generic.List[string]
foreach ($relativeName in $fileMap.Keys) {
    $templateName = $fileMap[$relativeName]
    $target = Join-Path $guidedDir $relativeName
    $content = Render-Template -Text (Get-TemplateText -TemplateName $templateName) -PaperTitle $paperTitle -PaperFolder $paperFolder -ScanDate $scanDate
    if (Write-IfNeeded -Target $target -Content $content -ShouldForce $Force.IsPresent) {
        [void]$created.Add((Join-Path '07_guided_reading_cn' $relativeName).Replace('\', '/'))
    }
}

if ($ChapterTemplate) {
    $chapterTarget = Join-Path $guidedDir 'Chapter01.md'
    $chapterContent = Render-Template -Text (Get-TemplateText -TemplateName 'guided_reading_chapter.md') -PaperTitle $paperTitle -PaperFolder $paperFolder -ScanDate $scanDate
    if (Write-IfNeeded -Target $chapterTarget -Content $chapterContent -ShouldForce $Force.IsPresent) {
        [void]$created.Add('07_guided_reading_cn/Chapter01.md')
    }
}

if ($created.Count -gt 0) {
    foreach ($item in $created) {
        Write-Output "[CREATED] $item"
    }
}
else {
    Write-Output '[SKIP] Guided-reading files already exist. Use -Force to overwrite.'
}
