package core

import (
	"container/list"
	"sync"
	"time"

	"github.com/elum-utils/censor/models"
)

const (
	B  int = 1
	KB     = 1024 * B
	MB     = 1024 * KB
	GB     = 1024 * MB
	TB     = 1024 * GB
	PB     = 1024 * TB
)

type negativeCacheEntry struct {
	key       string
	value     models.AIResult
	expiresAt time.Time
	sizeBytes int
}

// negativeResultCache is an in-memory LRU cache with TTL.
type negativeResultCache struct {
	mu         sync.Mutex
	maxBytes   int64
	totalBytes int64
	items      map[string]*list.Element
	lru        *list.List
}

func newNegativeResultCache(maxBytes int64) *negativeResultCache {
	if maxBytes <= 0 {
		return nil
	}
	return &negativeResultCache{
		maxBytes: maxBytes,
		items:    make(map[string]*list.Element),
		lru:      list.New(),
	}
}

func (c *negativeResultCache) Get(key string, now time.Time) (models.AIResult, bool) {
	if c == nil || key == "" {
		return models.AIResult{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	elem, ok := c.items[key]
	if !ok {
		return models.AIResult{}, false
	}
	entry := elem.Value.(*negativeCacheEntry)
	if now.After(entry.expiresAt) {
		c.removeElement(elem)
		return models.AIResult{}, false
	}
	c.lru.MoveToFront(elem)
	return entry.value, true
}

func (c *negativeResultCache) Set(key string, value models.AIResult, ttl time.Duration, now time.Time) {
	if c == nil || key == "" || ttl <= 0 {
		return
	}
	expiresAt := now.Add(ttl)
	newSize := estimateEntrySizeBytes(key, value)
	if int64(newSize) > c.maxBytes {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if elem, ok := c.items[key]; ok {
		entry := elem.Value.(*negativeCacheEntry)
		c.totalBytes -= int64(entry.sizeBytes)
		entry.value = value
		entry.expiresAt = expiresAt
		entry.sizeBytes = newSize
		c.totalBytes += int64(newSize)
		c.lru.MoveToFront(elem)
		c.evictToFitLocked()
		return
	}

	entry := &negativeCacheEntry{
		key:       key,
		value:     value,
		expiresAt: expiresAt,
		sizeBytes: newSize,
	}
	elem := c.lru.PushFront(entry)
	c.items[key] = elem
	c.totalBytes += int64(newSize)
	c.evictToFitLocked()
}

func (c *negativeResultCache) RemoveExpired(now time.Time) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for elem := c.lru.Back(); elem != nil; {
		prev := elem.Prev()
		entry := elem.Value.(*negativeCacheEntry)
		if now.After(entry.expiresAt) {
			c.removeElement(elem)
		}
		elem = prev
	}
}

func (c *negativeResultCache) removeElement(elem *list.Element) {
	if elem == nil {
		return
	}
	entry := elem.Value.(*negativeCacheEntry)
	delete(c.items, entry.key)
	c.lru.Remove(elem)
	c.totalBytes -= int64(entry.sizeBytes)
	if c.totalBytes < 0 {
		c.totalBytes = 0
	}
}

func (c *negativeResultCache) evictToFitLocked() {
	for c.totalBytes > c.maxBytes && c.lru.Len() > 0 {
		c.removeElement(c.lru.Back())
	}
}

func estimateEntrySizeBytes(key string, value models.AIResult) int {
	size := len(key)
	size += len(value.Reason)
	for _, token := range value.TriggerTokens {
		size += len(token)
	}
	// Approximate scalar/object overhead.
	size += 128
	return size
}
