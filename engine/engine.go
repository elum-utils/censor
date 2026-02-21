package engine

import (
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
)

// Stats contains runtime in-memory engine metrics.
type Stats struct {
	TokenCount       int64
	LastLookupNanos  int64
	TotalLookups     int64
	TotalTokenHits   int64
	LastReloadNanos  int64
	TotalReloadCount int64
}

type state struct {
	tokens  map[string]struct{}
	phrases []string
}

// Engine stores trigger tokens and executes case-insensitive lookup.
type Engine struct {
	mu    sync.RWMutex
	state state

	lastLookupNanos atomic.Int64
	totalLookups    atomic.Int64
	totalTokenHits  atomic.Int64
	lastReloadNanos atomic.Int64
	totalReloads    atomic.Int64
}

// New creates a new engine.
func New() *Engine {
	return &Engine{state: state{tokens: make(map[string]struct{})}}
}

func normalizeToken(token string) string {
	return strings.ToLower(strings.TrimSpace(token))
}

// AddToken inserts one token.
func (e *Engine) AddToken(token string) bool {
	t := normalizeToken(token)
	if t == "" {
		return false
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if _, exists := e.state.tokens[t]; exists {
		return false
	}
	e.state.tokens[t] = struct{}{}
	if strings.ContainsRune(t, ' ') {
		e.state.phrases = append(e.state.phrases, t)
	}
	return true
}

// RemoveToken deletes one token.
func (e *Engine) RemoveToken(token string) bool {
	t := normalizeToken(token)
	if t == "" {
		return false
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if _, exists := e.state.tokens[t]; !exists {
		return false
	}
	delete(e.state.tokens, t)
	if strings.ContainsRune(t, ' ') {
		phrases := e.state.phrases[:0]
		for _, p := range e.state.phrases {
			if p != t {
				phrases = append(phrases, p)
			}
		}
		e.state.phrases = phrases
	}
	return true
}

// ReplaceAll replaces all tokens atomically.
func (e *Engine) ReplaceAll(tokens []string) {
	start := time.Now()
	next := state{tokens: make(map[string]struct{}, len(tokens))}
	for _, token := range tokens {
		t := normalizeToken(token)
		if t == "" {
			continue
		}
		if _, exists := next.tokens[t]; exists {
			continue
		}
		next.tokens[t] = struct{}{}
		if strings.ContainsRune(t, ' ') {
			next.phrases = append(next.phrases, t)
		}
	}

	e.mu.Lock()
	e.state = next
	e.mu.Unlock()

	e.lastReloadNanos.Store(time.Since(start).Nanoseconds())
	e.totalReloads.Add(1)
}

// Clear removes all tokens.
func (e *Engine) Clear() {
	e.mu.Lock()
	e.state = state{tokens: make(map[string]struct{})}
	e.mu.Unlock()
}

// Count returns token count.
func (e *Engine) Count() int {
	e.mu.RLock()
	count := len(e.state.tokens)
	e.mu.RUnlock()
	return count
}

// FindTriggers returns unique tokens found in the message.
func (e *Engine) FindTriggers(message string) []string {
	start := time.Now()
	lower := strings.ToLower(message)
	e.mu.RLock()
	if len(e.state.tokens) == 0 || lower == "" {
		e.mu.RUnlock()
		e.lastLookupNanos.Store(time.Since(start).Nanoseconds())
		e.totalLookups.Add(1)
		return nil
	}

	found := make(map[string]struct{}, 4)

	// First pass: word-level exact matches.
	for _, tok := range splitTokens(lower) {
		if _, ok := e.state.tokens[tok]; ok {
			found[tok] = struct{}{}
		}
	}

	// Second pass: multi-word phrases.
	for _, phrase := range e.state.phrases {
		if _, already := found[phrase]; already {
			continue
		}
		if strings.Contains(lower, phrase) {
			found[phrase] = struct{}{}
		}
	}
	e.mu.RUnlock()

	if len(found) == 0 {
		e.lastLookupNanos.Store(time.Since(start).Nanoseconds())
		e.totalLookups.Add(1)
		return nil
	}

	out := make([]string, 0, len(found))
	for token := range found {
		out = append(out, token)
	}

	e.totalTokenHits.Add(int64(len(out)))
	e.lastLookupNanos.Store(time.Since(start).Nanoseconds())
	e.totalLookups.Add(1)
	return out
}

func splitTokens(s string) []string {
	res := make([]string, 0, 16)
	start := -1
	for i, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			if start == -1 {
				start = i
			}
			continue
		}
		if start != -1 {
			res = append(res, s[start:i])
			start = -1
		}
	}
	if start != -1 {
		res = append(res, s[start:])
	}
	return res
}

// Stats returns current metrics.
func (e *Engine) Stats() Stats {
	return Stats{
		TokenCount:       int64(e.Count()),
		LastLookupNanos:  e.lastLookupNanos.Load(),
		TotalLookups:     e.totalLookups.Load(),
		TotalTokenHits:   e.totalTokenHits.Load(),
		LastReloadNanos:  e.lastReloadNanos.Load(),
		TotalReloadCount: e.totalReloads.Load(),
	}
}
