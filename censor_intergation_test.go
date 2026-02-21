package censor_test

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elum-utils/censor"
	"github.com/elum-utils/censor/adapters/ai"
	"github.com/elum-utils/censor/adapters/storage"
	"github.com/elum-utils/censor/models"
)

// Manual integration test with real DeepSeek.
//
// Required env:
//
//	CENSOR_IT_DEEPSEEK_API_KEY
//
// Optional env:
//
//	CENSOR_IT_DEEPSEEK_BASE_URL (default https://api.deepseek.com)
//	CENSOR_IT_DEEPSEEK_MODEL    (default deepseek-chat)
func TestCensorIntergation_SQLiteRealDeepSeek(t *testing.T) {
	loadDotEnv(".env")

	apiKey := strings.TrimSpace(os.Getenv("CENSOR_IT_DEEPSEEK_API_KEY"))
	if apiKey == "" {
		t.Skip("set CENSOR_IT_DEEPSEEK_API_KEY to run real integration test")
	}

	baseURL := strings.TrimSpace(os.Getenv("CENSOR_IT_DEEPSEEK_BASE_URL"))
	if baseURL == "" {
		baseURL = "https://api.deepseek.com"
	}
	model := strings.TrimSpace(os.Getenv("CENSOR_IT_DEEPSEEK_MODEL"))
	if model == "" {
		model = "deepseek-chat"
	}
	const triggerToken = "bad"
	const triggerMessage = "this is bad"

	registerSQLiteStub()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	defer db.Close()

	st, err := storage.NewSQLAdapter(db, "censor_tokens")
	if err != nil {
		t.Fatalf("new sql adapter: %v", err)
	}
	if err := st.EnsureSchema(context.Background()); err != nil {
		t.Fatalf("ensure schema: %v", err)
	}
	if err := st.AddToken(context.Background(), strings.ToLower(triggerToken)); err != nil {
		t.Fatalf("seed token: %v", err)
	}

	realAI, err := ai.NewDeepSeekAdapter(ai.DeepSeekOptions{
		APIKey:  apiKey,
		BaseURL: baseURL,
		Model:   model,
		Timeout: 20 * time.Second,
	})
	if err != nil {
		t.Fatalf("new deepseek adapter: %v", err)
	}

	cb := &callbackCounter{}
	f := censor.New(censor.Options{
		AIAnalyzer:          realAI,
		Storage:             st,
		CallbackHandler:     cb,
		AutoLearn:           false,
		ConfidenceThreshold: 0.99,
		SyncInterval:        time.Hour,
	})
	if err := f.SyncOnce(context.Background()); err != nil {
		t.Fatalf("sync: %v", err)
	}

	// Short sanity path: no trigger -> clean callback, AI should be skipped.
	v1, err := f.ProcessMessage(context.Background(), models.Message{ID: 1, User: 101, Data: "hello"})
	if err != nil {
		t.Fatalf("process clean: %v", err)
	}
	if v1.AIResult.StatusCode != models.StatusClean {
		t.Fatalf("expected clean status, got %d", v1.AIResult.StatusCode)
	}
	if cb.clean.Load() != 1 {
		t.Fatalf("clean callback must be called once")
	}

	// Trigger path: seeded token should force AI call and one of status callbacks.
	beforeTotal := cb.total()
	v2, err := f.ProcessMessage(context.Background(), models.Message{ID: 2, User: 202, Data: triggerMessage})
	if err != nil {
		if isNetworkUnavailableError(err) {
			t.Skipf("deepseek network unavailable in current environment: %v", err)
		}
		t.Fatalf("process trigger: %v", err)
	}
	if !v2.Triggered {
		t.Fatalf("expected trigger detection")
	}
	if !v2.AIResult.StatusCode.Valid() {
		t.Fatalf("unexpected status from AI: %d", v2.AIResult.StatusCode)
	}
	afterTotal := cb.total()
	if afterTotal-beforeTotal != 1 {
		t.Fatalf("expected exactly one callback for trigger message, got delta=%d", afterTotal-beforeTotal)
	}
}

type callbackCounter struct {
	clean      atomic.Int64
	abuse      atomic.Int64
	suspicious atomic.Int64
	commercial atomic.Int64
	dangerous  atomic.Int64
	critical   atomic.Int64
}

func (c *callbackCounter) totalNonClean() int64 {
	return c.abuse.Load() + c.suspicious.Load() + c.commercial.Load() + c.dangerous.Load() + c.critical.Load()
}

func (c *callbackCounter) total() int64 {
	return c.clean.Load() + c.totalNonClean()
}

func (c *callbackCounter) OnClean(context.Context, models.Violation) error {
	c.clean.Add(1)
	return nil
}
func (c *callbackCounter) OnNonCriticalAbuse(context.Context, models.Violation) error {
	c.abuse.Add(1)
	return nil
}
func (c *callbackCounter) OnSuspicious(context.Context, models.Violation) error {
	c.suspicious.Add(1)
	return nil
}
func (c *callbackCounter) OnCommercialOffPlatform(context.Context, models.Violation) error {
	c.commercial.Add(1)
	return nil
}
func (c *callbackCounter) OnDangerousIllegal(context.Context, models.Violation) error {
	c.dangerous.Add(1)
	return nil
}
func (c *callbackCounter) OnCritical(context.Context, models.Violation) error {
	c.critical.Add(1)
	return nil
}

var sqliteRegisterOnce sync.Once

func registerSQLiteStub() {
	sqliteRegisterOnce.Do(func() {
		sql.Register("sqlite", &sqliteStubDriver{store: &sqliteStubStore{tokens: make(map[string]struct{})}})
	})
}

type sqliteStubStore struct {
	mu     sync.Mutex
	tokens map[string]struct{}
}

type sqliteStubDriver struct{ store *sqliteStubStore }

type sqliteStubConn struct{ store *sqliteStubStore }

type sqliteStubRows struct {
	data []string
	idx  int
}

type sqliteStubResult struct{}

func (d *sqliteStubDriver) Open(string) (driver.Conn, error) {
	return &sqliteStubConn{store: d.store}, nil
}

func (c *sqliteStubConn) Prepare(string) (driver.Stmt, error) { return nil, fmt.Errorf("not used") }
func (c *sqliteStubConn) Close() error                        { return nil }
func (c *sqliteStubConn) Begin() (driver.Tx, error)           { return nil, fmt.Errorf("not used") }

func (c *sqliteStubConn) ExecContext(_ context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	q := strings.ToLower(query)
	c.store.mu.Lock()
	defer c.store.mu.Unlock()
	switch {
	case strings.Contains(q, "create table"):
		return sqliteStubResult{}, nil
	case strings.Contains(q, "insert"):
		token := fmt.Sprint(args[0].Value)
		if _, exists := c.store.tokens[token]; exists {
			return nil, fmt.Errorf("duplicate")
		}
		c.store.tokens[token] = struct{}{}
		return sqliteStubResult{}, nil
	case strings.Contains(q, "delete"):
		token := fmt.Sprint(args[0].Value)
		delete(c.store.tokens, token)
		return sqliteStubResult{}, nil
	default:
		return nil, fmt.Errorf("unsupported exec")
	}
}

func (c *sqliteStubConn) QueryContext(_ context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	q := strings.ToLower(query)
	c.store.mu.Lock()
	defer c.store.mu.Unlock()
	if strings.Contains(q, "limit 1") {
		token := fmt.Sprint(args[0].Value)
		if _, ok := c.store.tokens[token]; !ok {
			return &sqliteStubRows{data: nil}, nil
		}
		return &sqliteStubRows{data: []string{"1"}}, nil
	}
	out := make([]string, 0, len(c.store.tokens))
	for token := range c.store.tokens {
		out = append(out, token)
	}
	return &sqliteStubRows{data: out}, nil
}

func (r *sqliteStubRows) Columns() []string { return []string{"token"} }
func (r *sqliteStubRows) Close() error      { return nil }
func (r *sqliteStubRows) Next(dest []driver.Value) error {
	if r.idx >= len(r.data) {
		return io.EOF
	}
	dest[0] = r.data[r.idx]
	r.idx++
	return nil
}

func (sqliteStubResult) LastInsertId() (int64, error) { return 0, nil }
func (sqliteStubResult) RowsAffected() (int64, error) { return 1, nil }

var _ driver.Driver = (*sqliteStubDriver)(nil)
var _ driver.Conn = (*sqliteStubConn)(nil)
var _ driver.ExecerContext = (*sqliteStubConn)(nil)
var _ driver.QueryerContext = (*sqliteStubConn)(nil)
var _ driver.Rows = (*sqliteStubRows)(nil)

func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	lines := strings.Split(string(data), "\n")
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		i := strings.IndexByte(line, '=')
		if i <= 0 {
			continue
		}
		key := strings.TrimSpace(line[:i])
		val := strings.TrimSpace(line[i+1:])
		val = strings.Trim(val, `"'`)
		if key == "" {
			continue
		}
		if os.Getenv(key) == "" {
			_ = os.Setenv(key, val)
		}
	}
}

func isNetworkUnavailableError(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "no such host") ||
		strings.Contains(s, "dial tcp") ||
		strings.Contains(s, "network is unreachable") ||
		strings.Contains(s, "connection refused") ||
		strings.Contains(s, "i/o timeout") ||
		strings.Contains(s, "timeout")
}
