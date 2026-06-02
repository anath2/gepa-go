package gepa

// proposalOutcome is the engine-level result of a mutation attempt (reflection, merge, …).
// Failed proposals are non-fatal: the loop records the reason and continues.
type proposalOutcome struct {
	Instruction     string
	RawResponseText string
	Candidate       Candidate
	Failed          bool
	Reason          string
}

func mutatedCandidate(parent Candidate, moduleName, instruction string) Candidate {
	proposal := cloneCandidate(parent)
	proposal[moduleName] = instruction
	return proposal
}
