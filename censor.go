package censor

import "github.com/elum-utils/censor/core"

// Re-export core API at module root for convenient imports.
type (
	Core           = core.Core
	Options        = core.Options
	ProcessOptions = core.ProcessOptions
	EventName      = core.EventName
	ViolationEvent = core.ViolationEvent
	EventHandler   = core.EventHandler
)

const (
	EventAllowClean       = core.EventAllowClean
	EventMarkAbuse        = core.EventMarkAbuse
	EventHumanReview      = core.EventHumanReview
	EventAutoRestrict     = core.EventAutoRestrict
	EventAutoBanEscalate  = core.EventAutoBanEscalate
	EventCriticalEscalate = core.EventCriticalEscalate
)

// New creates a new content safety filter.
func New(opt Options) *Core {
	return core.New(opt)
}
