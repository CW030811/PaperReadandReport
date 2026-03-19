param(
    [Parameter(Mandatory = $true)]
    [string]$Title,
    [string]$Date = (Get-Date -Format 'yyyy-MM-dd'),
    [string]$Root = 'papers',
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:RepoRoot = Split-Path -Parent $PSScriptRoot
$script:TemplateDir = Join-Path $script:RepoRoot 'templates'

$subdirs = @(
    '00_pdf',
    '01_source_text',
    '02_notes',
    '03_extraction',
    '04_figures',
    '05_slides',
    '06_compare',
    '07_guided_reading_cn'
)

$fileMap = [ordered]@{
    '02_notes/deep_notes.md' = 'deep_notes.md'
    '03_extraction/paper_brief.md' = 'paper_brief.md'
    '03_extraction/paper_schema.json' = 'paper_schema.json'
    '05_slides/slides_outline.md' = 'slides_outline.md'
    '06_compare/comparison_table.md' = 'comparison_table.md'
    '07_guided_reading_cn/reading_map.md' = 'guided_reading_map.md'
    '07_guided_reading_cn/guided_reading_rules.md' = 'guided_reading_rules.md'
    '07_guided_reading_cn/progress_log.md' = 'guided_reading_progress_log.md'
}

function Get-Slug {
    param([string]$Value)
    $slug = $Value.Trim().ToLowerInvariant()
    $slug = [regex]::Replace($slug, '[^a-z0-9]+', '-')
    $slug = [regex]::Replace($slug, '-{2,}', '-').Trim('-')
    if ([string]::IsNullOrWhiteSpace($slug)) {
        return 'untitled-paper'
    }
    return $slug
}

function Get-TemplateText {
    param([string]$TemplateName)
    return Get-Content -Raw -Encoding UTF8 (Join-Path $script:TemplateDir $TemplateName)
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
    $directory = Split-Path -Parent $Path
    if ($directory) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

$workspaceName = '{0}_{1}' -f $Date, (Get-Slug -Value $Title)
$workspaceDir = Join-Path (Join-Path $script:RepoRoot $Root) $workspaceName

if ((Test-Path $workspaceDir) -and -not $Force) {
    throw "Workspace already exists: $workspaceDir`nUse -Force to reuse it."
}

New-Item -ItemType Directory -Force -Path $workspaceDir | Out-Null
foreach ($subdir in $subdirs) {
    $currentDir = Join-Path $workspaceDir $subdir
    New-Item -ItemType Directory -Force -Path $currentDir | Out-Null
    if ($subdir -in @('00_pdf', '01_source_text', '04_figures')) {
        $gitkeep = Join-Path $currentDir '.gitkeep'
        if (-not (Test-Path $gitkeep)) {
            New-Item -ItemType File -Path $gitkeep | Out-Null
        }
    }
}

$paperFolder = (Join-Path $Root $workspaceName).Replace('\', '/')
foreach ($relativePath in $fileMap.Keys) {
    $templateName = $fileMap[$relativePath]
    $target = Join-Path $workspaceDir ($relativePath -replace '/', '\')
    if ($templateName -eq 'paper_schema.json') {
        $schema = Get-TemplateText -TemplateName $templateName | ConvertFrom-Json
        $schema.paper_title = $Title
        $schema.paper_folder = $paperFolder
        $schema.scan_date = $Date
        Write-Utf8File -Path $target -Content (($schema | ConvertTo-Json -Depth 8) + "`n")
    }
    else {
        $content = Render-Template -Text (Get-TemplateText -TemplateName $templateName) -PaperTitle $Title -PaperFolder $paperFolder -ScanDate $Date
        Write-Utf8File -Path $target -Content $content
    }
}

Write-Output $workspaceDir
