package storage

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
)

func TestMemoryAdapter(t *testing.T) {
	m := NewMemoryAdapter()
	ctx := context.Background()
	_ = m.AddToken(ctx, "a")
	ok, _ := m.TokenExists(ctx, "a")
	if !ok {
		t.Fatalf("expected token")
	}
	all, _ := m.GetTokens(ctx)
	if len(all) != 1 {
		t.Fatalf("unexpected size: %d", len(all))
	}
	_ = m.RemoveToken(ctx, "a")
	ok, _ = m.TokenExists(ctx, "a")
	if ok {
		t.Fatalf("expected token removed")
	}
}

func TestNewSQLAdapterNilDB(t *testing.T) {
	if _, err := NewSQLAdapter(nil, "t"); err == nil {
		t.Fatalf("expected error")
	}
}

func TestSQLAdapterWithStubDriver(t *testing.T) {
	driverName := "censor_stub_sql"
	sql.Register(driverName, &stubDriver{store: &stubStore{tokens: make(map[string]struct{})}})
	db, err := sql.Open(driverName, "")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	a, err := NewSQLAdapter(db, "tokens")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := a.EnsureSchema(ctx); err != nil {
		t.Fatal(err)
	}
	if err := a.AddToken(ctx, "x"); err != nil {
		t.Fatal(err)
	}
	if err := a.AddToken(ctx, "x"); err != nil {
		t.Fatal(err)
	}
	ok, err := a.TokenExists(ctx, "x")
	if err != nil || !ok {
		t.Fatalf("expected token exists: ok=%v err=%v", ok, err)
	}
	all, err := a.GetTokens(ctx)
	if err != nil || len(all) != 1 {
		t.Fatalf("unexpected tokens: %v err=%v", all, err)
	}
	if err := a.RemoveToken(ctx, "x"); err != nil {
		t.Fatal(err)
	}
	ok, err = a.TokenExists(ctx, "x")
	if err != nil || ok {
		t.Fatalf("expected token removed: ok=%v err=%v", ok, err)
	}
}

type stubStore struct {
	mu     sync.Mutex
	tokens map[string]struct{}
}

type stubDriver struct{ store *stubStore }

type stubConn struct{ store *stubStore }

type stubRows struct {
	data []string
	idx  int
}

type stubResult struct{}

func (d *stubDriver) Open(string) (driver.Conn, error) { return &stubConn{store: d.store}, nil }

func (c *stubConn) Prepare(string) (driver.Stmt, error) { return nil, errors.New("not used") }
func (c *stubConn) Close() error                        { return nil }
func (c *stubConn) Begin() (driver.Tx, error)           { return nil, errors.New("not used") }

func (c *stubConn) ExecContext(_ context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	q := strings.ToLower(query)
	c.store.mu.Lock()
	defer c.store.mu.Unlock()
	switch {
	case strings.Contains(q, "create table"):
		return stubResult{}, nil
	case strings.Contains(q, "insert"):
		token := fmt.Sprint(args[0].Value)
		if _, ok := c.store.tokens[token]; ok {
			return nil, errors.New("duplicate")
		}
		c.store.tokens[token] = struct{}{}
		return stubResult{}, nil
	case strings.Contains(q, "delete"):
		token := fmt.Sprint(args[0].Value)
		delete(c.store.tokens, token)
		return stubResult{}, nil
	default:
		return nil, errors.New("unsupported exec")
	}
}

func (c *stubConn) QueryContext(_ context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	q := strings.ToLower(query)
	c.store.mu.Lock()
	defer c.store.mu.Unlock()
	if strings.Contains(q, "limit 1") {
		token := fmt.Sprint(args[0].Value)
		if _, ok := c.store.tokens[token]; !ok {
			return &stubRows{data: nil}, nil
		}
		return &stubRows{data: []string{"1"}}, nil
	}
	out := make([]string, 0, len(c.store.tokens))
	for token := range c.store.tokens {
		out = append(out, token)
	}
	return &stubRows{data: out}, nil
}

func (r *stubRows) Columns() []string { return []string{"token"} }
func (r *stubRows) Close() error      { return nil }
func (r *stubRows) Next(dest []driver.Value) error {
	if r.idx >= len(r.data) {
		return io.EOF
	}
	dest[0] = r.data[r.idx]
	r.idx++
	return nil
}

func (stubResult) LastInsertId() (int64, error) { return 0, nil }
func (stubResult) RowsAffected() (int64, error) { return 1, nil }

var _ driver.Driver = (*stubDriver)(nil)
var _ driver.Conn = (*stubConn)(nil)
var _ driver.ExecerContext = (*stubConn)(nil)
var _ driver.QueryerContext = (*stubConn)(nil)
var _ driver.Rows = (*stubRows)(nil)
