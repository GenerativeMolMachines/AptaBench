# Автоматический коммит и push изменений
# Использование: .\scripts\auto_commit_push.ps1

$ErrorActionPreference = "Stop"
$repoPath = Split-Path -Parent $PSScriptRoot

# Переходим в директорию репозитория
Set-Location $repoPath

# Проверяем, что мы в git репозитории
if (-not (Test-Path ".git")) {
    Write-Error "Не найден git репозиторий в $repoPath"
    exit 1
}

# Получаем статус
$status = git status --porcelain

# Если нет изменений, выходим
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): Нет изменений для коммита" -ForegroundColor Gray
    exit 0
}

# Получаем имя текущей ветки
$branch = git rev-parse --abbrev-ref HEAD

Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): Обнаружены изменения на ветке $branch" -ForegroundColor Yellow

# Добавляем все изменения
git add -A

# Создаем коммит с временной меткой
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$commitMessage = "Auto-commit: $timestamp"

try {
    git commit -m $commitMessage
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): Коммит создан успешно" -ForegroundColor Green
    
    # Делаем push
    git push origin $branch
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): Push выполнен успешно" -ForegroundColor Green
} catch {
    Write-Error "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): Ошибка при коммите или push: $_"
    exit 1
}

