# Автоматические коммиты и push

Этот скрипт позволяет автоматически коммитить и пушить изменения в репозиторий через заданные интервалы времени.

## Настройка через Windows Task Scheduler

### Шаг 1: Открыть Task Scheduler
1. Нажмите `Win + R`
2. Введите `taskschd.msc` и нажмите Enter

### Шаг 2: Создать новую задачу
1. В правой панели нажмите "Create Basic Task" или "Create Task"
2. Назовите задачу (например, "Auto Git Commit and Push")
3. Выберите "When I log on" или "When the computer starts" для триггера

### Шаг 3: Настроить триггер
1. Перейдите на вкладку "Triggers"
2. Нажмите "New"
3. Выберите "On a schedule"
4. Выберите частоту (ежедневно, еженедельно, или по расписанию)
5. Установите время и частоту повторения (например, каждые 2 часа)

### Шаг 4: Настроить действие
1. Перейдите на вкладку "Actions"
2. Нажмите "New"
3. В "Program/script" введите: `powershell.exe`
4. В "Add arguments" введите: `-ExecutionPolicy Bypass -File "C:\Users\m19er\AptaBench\scripts\auto_commit_push.ps1"`
5. В "Start in" введите: `C:\Users\m19er\AptaBench`

### Шаг 5: Дополнительные настройки
1. Перейдите на вкладку "General"
2. Отметьте "Run whether user is logged on or not" (если нужно)
3. Отметьте "Run with highest privileges" (может потребоваться для доступа к файлам)

### Шаг 6: Сохранить задачу
Нажмите "OK" и введите пароль администратора, если потребуется.

## Альтернативный способ: Использование PowerShell для создания задачи

Можно также создать задачу через PowerShell командой:

```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"C:\Users\m19er\AptaBench\scripts\auto_commit_push.ps1`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Days 365)
Register-ScheduledTask -TaskName "AutoGitCommitPush" -Action $action -Trigger $trigger -Description "Автоматический коммит и push изменений в Git"
```

## Важные замечания

⚠️ **Внимание**: Автоматические коммиты могут быть не лучшей практикой:
- Они коммитят все изменения без проверки
- Могут создать "шум" в истории коммитов
- Могут закоммитить временные файлы или файлы с ошибками

### Рекомендации:
1. Убедитесь, что `.gitignore` правильно настроен
2. Рассмотрите возможность коммитить только определенные типы файлов
3. Используйте автоматические коммиты только для резервного копирования, а не для основной работы

## Проверка работы

Чтобы проверить, что задача работает:
1. Откройте Task Scheduler
2. Найдите вашу задачу
3. Нажмите правой кнопкой → "Run" для тестового запуска
4. Проверьте логи в "History" вкладке

## Остановка автоматических коммитов

Чтобы остановить автоматические коммиты:
1. Откройте Task Scheduler
2. Найдите вашу задачу
3. Нажмите правой кнопкой → "Disable" или "Delete"

