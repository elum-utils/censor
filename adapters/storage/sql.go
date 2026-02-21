package storage

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// SQLAdapter is a generic SQL storage implementation.
type SQLAdapter struct {
	db    *sql.DB
	table string
}

// NewSQLAdapter creates an adapter over *sql.DB.
func NewSQLAdapter(db *sql.DB, table string) (*SQLAdapter, error) {
	if db == nil {
		return nil, errors.New("storage: db is nil")
	}
	if strings.TrimSpace(table) == "" {
		table = "censor_tokens"
	}
	return &SQLAdapter{db: db, table: table}, nil
}

// EnsureSchema creates table if missing.
func (s *SQLAdapter) EnsureSchema(ctx context.Context) error {
	q := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (token TEXT PRIMARY KEY)`, s.table)
	_, err := s.db.ExecContext(ctx, q)
	return err
}

func (s *SQLAdapter) AddToken(ctx context.Context, token string) error {
	q := fmt.Sprintf(`INSERT INTO %s (token) VALUES (?)`, s.table)
	_, err := s.db.ExecContext(ctx, q, token)
	if err == nil {
		return nil
	}
	if strings.Contains(strings.ToLower(err.Error()), "duplicate") || strings.Contains(strings.ToLower(err.Error()), "unique") {
		return nil
	}
	return err
}

func (s *SQLAdapter) RemoveToken(ctx context.Context, token string) error {
	q := fmt.Sprintf(`DELETE FROM %s WHERE token = ?`, s.table)
	_, err := s.db.ExecContext(ctx, q, token)
	return err
}

func (s *SQLAdapter) GetTokens(ctx context.Context) ([]string, error) {
	q := fmt.Sprintf(`SELECT token FROM %s`, s.table)
	rows, err := s.db.QueryContext(ctx, q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]string, 0, 256)
	for rows.Next() {
		var token string
		if scanErr := rows.Scan(&token); scanErr != nil {
			return nil, scanErr
		}
		out = append(out, token)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *SQLAdapter) TokenExists(ctx context.Context, token string) (bool, error) {
	q := fmt.Sprintf(`SELECT 1 FROM %s WHERE token = ? LIMIT 1`, s.table)
	var v int
	err := s.db.QueryRowContext(ctx, q, token).Scan(&v)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}
