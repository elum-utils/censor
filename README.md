# censor

`github.com/elum-utils/censor` — пакет многоуровневой модерации контента на Go:

1. Быстрый in-memory pre-filter (case-insensitive, low-allocation).
2. AI-анализатор через адаптеры (DeepSeek/OpenAI-compatible и т.д.).
3. Автообучение trigger-токенов (слова и фразы).
4. Callback-события по статусам 1..6.
5. LRU+TTL in-memory кеш результатов AI для повторяющихся сообщений.

## Структура

- `core` — основная логика.
- `engine` — потокобезопасный in-memory движок триггеров.
- `models` — сообщения и результаты AI (компактный и полный JSON).
- `interfaces` — интерфейсы AI/Storage/Callback/Logger.
- `adapters/ai` — AI-адаптеры.
- `adapters/storage` — Storage-адаптеры.

## Статусы

- `1` clean
- `2` non-critical abuse
- `3` human review required
- `4` suspicious (competitors, moving users to other apps because it's better)
- `5` commercial/off-platform (selling intimate content/services, etc.)
- `6` dangerous/illegal

## Форматы AI-ответа

Компактный:

```json
{"a":4,"b":"reason","c":0.93,"d":["token"],"e":123,"f":777}
```

Минимальный компактный (рекомендуется для экономии токенов):

```json
{"a":4,"f":777,"c":0.93,"d":["token"]}
```

Где:
- `a` — `status_code`
- `f` — `message_id`
- `c` — `confidence`
- `d` — `trigger_tokens`

Полный:

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

	"github.com/elum-utils/censor"
	"github.com/elum-utils/censor/adapters/ai"
	"github.com/elum-utils/censor/adapters/storage"
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
			// SystemPrompt: "custom prompt ...", // optional
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
			MaxLearnTokenLength: 255,
			CacheTTL:            1 * time.Hour,
			CacheMaxBytes:       32 * censor.MB,
		})

		_ = c.OnAllowClean(func(ctx context.Context, e censor.ViolationEvent) error {
			fmt.Printf("[CENSOR][%s] msg=%d user=%d reason=%s\n", censor.EventAllowClean, e.MessageID, e.ViolatorUserID, e.Reason)
			return nil
		})
		_ = c.OnMarkAbuse(func(ctx context.Context, e censor.ViolationEvent) error { return nil })
		_ = c.OnHumanReview(func(ctx context.Context, e censor.ViolationEvent) error { return nil })
		_ = c.OnAutoRestrict(func(ctx context.Context, e censor.ViolationEvent) error { return nil })
		_ = c.OnAutoBanEscalate(func(ctx context.Context, e censor.ViolationEvent) error { return nil })
		_ = c.OnCriticalEscalate(func(ctx context.Context, e censor.ViolationEvent) error { return nil })

		return c.Run(ctx)
	}
}
```

## Обработка с опциями

Можно принудительно обходить pre-filter по триггерам:

```go
res, err := c.ProcessMessageWithOptions(ctx, msg, censor.ProcessOptions{
	SkipTriggerFilter: true,
})
```

Есть batch-вариант:

```go
res, err := c.ProcessBatchWithOptions(ctx, messages, censor.ProcessOptions{
	SkipTriggerFilter: true,
})
```

## Кеш AI-результатов

- Ключ кеша: текст сообщения (`message.Data` после trim по `MaxMessageSize`).
- Значение: полный `AIResult` + TTL.
- Кеш: LRU + TTL, фоновая очистка в горутине.
- При cache-hit AI не вызывается.
- В cache-hit сохраняется исходное решение (`status/reason/confidence/tokens`), но подставляются текущие `MessageID` и `ViolatorUserID`.
- Работает и в batch: для каждого сообщения проверяется кеш отдельно.

## Batch формат для AI

По умолчанию AI получает массив:

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
