# censor

`github.com/elum-utils/censor` — пакет многоуровневой модерации контента на Go:

1. Быстрый in-memory pre-filter (case-insensitive).
2. AI-анализ через адаптеры (DeepSeek/OpenAI-compatible и т.д.).
3. Callback-события по статусам `1..6`.
4. LRU+TTL in-memory кеш результатов AI для повторяющихся сообщений.
5. Автообучение trigger-токенов (только для уровней `4..6`).

## Структура

- `core` — основная логика и публичный API.
- `engine` — in-memory движок триггеров.
- `models` — сообщения и результаты AI.
- `interfaces` — интерфейсы AI/Storage/Callback/Logger.
- `adapters/ai` — AI-адаптеры.
- `adapters/storage` — Storage-адаптеры.

## Статусы

- `1` clean
- `2` non-critical abuse
- `3` human review required
- `4` suspicious (конкуренты, переманивание в другие приложения)
- `5` commercial/off-platform (продажа интима/услуг и т.д.)
- `6` dangerous/illegal

## Форматы AI-ответа

Рекомендуемый компактный формат:

```json
{"a":4,"f":777,"c":0.93,"d":["token"]}
```

Где:
- `a` — `status_code`
- `f` — `message_id`
- `c` — `confidence`
- `d` — `trigger_tokens`

Поддерживаются также расширенные поля (`b`, `e`) и полный формат для обратной совместимости.

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
			fmt.Printf("[CENSOR][%s] msg=%d user=%d cache=%t\n", censor.EventAllowClean, e.MessageID, e.ViolatorUserID, e.CacheHit)
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

Batch-вариант:

```go
res, err := c.ProcessBatchWithOptions(ctx, messages, censor.ProcessOptions{
	SkipTriggerFilter: true,
})
```

## Кеш AI-результатов

- Ключ: текст сообщения (`message.Data` после ограничения `MaxMessageSize`).
- Значение: `AIResult` + TTL.
- Политика: LRU + TTL, фоновая очистка в горутине.
- При cache-hit AI не вызывается.
- В cache-hit используется исходное решение AI, но подставляются текущие `MessageID` и `ViolatorUserID`.
- В `ViolationEvent` есть явный признак `CacheHit`.
- Работает и в batch: каждое сообщение проверяется отдельно.

## Обучение токенов

- Автообучение работает только для уровней `4..6`.
- Для `1..3` trigger-токены от AI можно не возвращать.
- Длина каждого токена/фразы ограничена `MaxLearnTokenLength` (по умолчанию 255).

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
