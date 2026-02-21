package engine

import (
	"sync"
	"testing"
)

func TestEngineFindTriggersCaseInsensitive(t *testing.T) {
	e := New()
	e.AddToken("BaD")
	e.AddToken("buy now")

	got := e.FindTriggers("This is BAD. Please BUY NOW!")
	if len(got) != 2 {
		t.Fatalf("expected 2 triggers, got %d: %v", len(got), got)
	}
}

func TestEngineReplaceAllAndClear(t *testing.T) {
	e := New()
	e.ReplaceAll([]string{"a", "b", "b", " "})
	if e.Count() != 2 {
		t.Fatalf("expected 2 tokens, got %d", e.Count())
	}
	e.Clear()
	if e.Count() != 0 {
		t.Fatalf("expected 0 after clear, got %d", e.Count())
	}
}

func TestEngineConcurrentAccess(t *testing.T) {
	e := New()
	e.AddToken("spam")

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = e.FindTriggers("SPAM spam")
			_ = e.AddToken("x")
			_ = e.RemoveToken("x")
		}()
	}
	wg.Wait()
}
