# censor

`github.com/elum-utils/censor` — пакет многоуровневой модерации контента на Go:

1. Быстрый in-memory pre-filter (case-insensitive, low-allocation).
2. AI-анализ только для сообщений с найденными триггерами.
3. Автообучение trigger-токенов по confidence.
4. Callback-события по статусам 1..6.

## Структура

- `core` — основной API (`New`, `Run`, `ProcessMessage`, `ProcessBatch`, `On`).
- `engine` — потокобезопасный in-memory движок.
- `models` — сообщения и результаты AI (компактный и полный JSON форматы).
- `interfaces` — интерфейсы AI/Storage/Callback/Logger.
- `adapters/ai` — DeepSeek(OpenAI-compatible) адаптер.
- `adapters/storage` — Memory + SQL адаптеры.

## Статусы

- `1` clean
- `2` non-critical abuse
- `3` suspicious
- `4` commercial/off-platform
- `5` dangerous/illegal
- `6` critical

## Форматы AI-ответа

Поддерживается компактный формат:

```json
{"a":4,"b":"reason","c":0.93,"d":["token"],"e":123,"f":777}
```

И полный формат:

```json
{"status_code":4,"reason":"reason","confidence":0.93,"trigger_tokens":["token"],"violator_user_id":123,"message_id":777}
```

## Пример интеграции

```go
package myservice

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/elum-utils/censor/adapters/ai"
	"github.com/elum-utils/censor/adapters/storage"
	"github.com/elum-utils/censor"
)

func Service(db *sql.DB, apiKey, model, baseURL string) func(ctx context.Context) error {
	return func(ctx context.Context) error {
		if db == nil {
			return fmt.Errorf("censor: MainDB is nil")
		}

		aiAdapter, err := ai.NewDeepSeekAdapter(ai.DeepSeekOptions{
			APIKey:  apiKey,
			Model:   model,
			BaseURL: baseURL,
		})
		if err != nil {
			return err
		}
		storageAdapter, err := storage.NewSQLAdapter(db, "censor_tokens")
		if err != nil {
			return err
		}

		c := censor.New(censor.Options{
			AIAnalyzer:          aiAdapter,
			Storage:             storageAdapter,
			SyncInterval:        1 * time.Minute,
			ConfidenceThreshold: 0.7,
			MaxMessageSize:      4 * 1024,
			MaxLearnTokenLength: 255, // AI может добавлять слова и фразы до 255 символов
		})

		_ = c.OnAllowClean(func(ctx context.Context, e censor.ViolationEvent) error {
			fmt.Printf("[CENSOR][%s] msg=%d user=%d reason=%s\n", censor.EventAllowClean, e.MessageID, e.ViolatorUserID, e.Reason)
			return nil
		})

		return c.Run(ctx)
	}
}
```

## Batch по умолчанию

AI получает массив сообщений в формате:

```json
[{"id":1,"user":2,"data":"text"}]
```

AI может вернуть один объект или массив нарушений.

## Тесты

```bash
go test ./...
go test -race ./...
go test ./... -cover
```
