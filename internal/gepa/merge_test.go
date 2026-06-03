package gepa

import (
	"context"
	"math/rand"
	"testing"

	"github.com/anath2/gepa-go/internal/program"
)

func TestMergeDesirableWhenDescendantsDiverge(t *testing.T) {
	ancestor := Candidate{"m1": "A"}
	i := Candidate{"m1": "A"}
	j := Candidate{"m1": "B"}
	if !mergeDesirable(ancestor, i, j) {
		t.Fatal("mergeDesirable = false, want true when one child matches ancestor and other differs")
	}
}

func TestMergeDesirableFalseWhenDescendantsIdentical(t *testing.T) {
	ancestor := Candidate{"m1": "A"}
	i := Candidate{"m1": "A"}
	j := Candidate{"m1": "A"}
	if mergeDesirable(ancestor, i, j) {
		t.Fatal("mergeDesirable = true, want false when descendants are identical")
	}
}

func TestFilterMergeAncestorsSkipsPreviouslyMerged(t *testing.T) {
	tracker := newMergeTracker()
	tracker.recordTriplet(1, 2, 0)

	scores := []float64{0.1, 0.5, 0.6}
	candidates := []candidateRecord{
		{ID: 0, Prompts: Candidate{"m1": "A"}},
		{ID: 1, Prompts: Candidate{"m1": "A"}},
		{ID: 2, Prompts: Candidate{"m1": "B"}},
	}

	got := filterMergeAncestors(tracker, 1, 2, []int{0}, scores, candidates)
	if len(got) != 0 {
		t.Fatalf("filterMergeAncestors = %v, want empty (already merged)", got)
	}
}

func TestFilterMergeAncestorsSkipsWhenAncestorOutscoresChild(t *testing.T) {
	tracker := newMergeTracker()
	scores := []float64{0.9, 0.5, 0.6}
	candidates := []candidateRecord{
		{ID: 0, Prompts: Candidate{"m1": "A"}},
		{ID: 1, Prompts: Candidate{"m1": "A"}},
		{ID: 2, Prompts: Candidate{"m1": "B"}},
	}

	got := filterMergeAncestors(tracker, 1, 2, []int{0}, scores, candidates)
	if len(got) != 0 {
		t.Fatalf("filterMergeAncestors = %v, want empty (ancestor outscores child)", got)
	}
}

func TestFindMergePairReturnsExpectedTriplet(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	tracker := newMergeTracker()
	state := poolState{
		Candidates: []candidateRecord{
			{ID: 0, ParentIDs: []int{}, Prompts: Candidate{"m1": "A"}},
			{ID: 1, ParentIDs: []int{0}, Prompts: Candidate{"m1": "A"}},
			{ID: 2, ParentIDs: []int{0}, Prompts: Candidate{"m1": "B"}},
		},
		TrainScores: [][]float64{
			{0, 0, 0, 0},
			{1, 1, 0, 0},
			{0, 0, 1, 1},
		},
	}

	pair, ok := findMergePair(state, tracker, rng, []int{1, 2})
	if !ok {
		t.Fatal("findMergePair = false, want true")
	}
	if pair.ID1 != 1 || pair.ID2 != 2 || pair.Ancestor != 0 {
		t.Fatalf("pair = %#v, want (1, 2, 0)", pair)
	}
}

func TestFindMergePairScansFrontierPairsExhaustively(t *testing.T) {
	const lastCandidate = 100
	tracker := newMergeTracker()
	candidates := []candidateRecord{
		{ID: 0, ParentIDs: []int{}, Prompts: Candidate{"m1": "A"}},
	}
	trainScores := [][]float64{{0.1}}
	mergeCandidates := make([]int, 0, lastCandidate)
	for id := 1; id <= lastCandidate; id++ {
		prompt := "A"
		if id == lastCandidate {
			prompt = "B"
		}
		candidates = append(candidates, candidateRecord{
			ID:        id,
			ParentIDs: []int{0},
			Prompts:   Candidate{"m1": prompt},
		})
		trainScores = append(trainScores, []float64{0.5})
		mergeCandidates = append(mergeCandidates, id)
	}
	for id := 1; id < lastCandidate-1; id++ {
		tracker.recordTriplet(id, lastCandidate, 0)
	}
	state := poolState{
		Candidates:  candidates,
		TrainScores: trainScores,
	}

	pair, ok := findMergePair(state, tracker, rand.New(rand.NewSource(0)), mergeCandidates)
	if !ok {
		t.Fatal("findMergePair = false, want exhaustive scan to find the only untried viable pair")
	}
	if pair.ID1 != lastCandidate-1 || pair.ID2 != lastCandidate || pair.Ancestor != 0 {
		t.Fatalf("pair = %#v, want (%d, %d, 0)", pair, lastCandidate-1, lastCandidate)
	}
}

func TestCombineMergedCandidateTakesDivergedModule(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	state := poolState{
		Candidates: []candidateRecord{
			{ID: 0, Prompts: Candidate{"m1": "A"}},
			{ID: 1, Prompts: Candidate{"m1": "A"}},
			{ID: 2, Prompts: Candidate{"m1": "B"}},
		},
		TrainScores: [][]float64{
			{0, 0},
			{0.6, 0.6},
			{0.7, 0.7},
		},
	}
	pair := mergePair{ID1: 1, ID2: 2, Ancestor: 0}

	merged := combineMergedCandidate(state, pair, rng)
	if merged["m1"] != "B" {
		t.Fatalf("merged[m1] = %q, want B", merged["m1"])
	}
}

func TestMergeTrackerRecordsAttemptedTriplet(t *testing.T) {
	tracker := newMergeTracker()
	tracker.recordTriplet(1, 2, 0)
	if !tracker.alreadyAttempted(1, 2, 0) {
		t.Fatal("alreadyAttempted = false, want true after recordTriplet")
	}
	if !tracker.alreadyAttempted(2, 1, 0) {
		t.Fatal("alreadyAttempted should treat (1,2) and (2,1) as the same triplet")
	}
}

func TestTryMergeIterationAcceptsCompatiblePair(t *testing.T) {
	prog := program.Program{
		Modules: []program.Module{
			{Name: "retriever", Prompt: "r0"},
			{Name: "answer", Prompt: "a0"},
		},
	}
	state := newPoolState(prog)
	train := []program.Example{
		{Input: map[string]any{"q": "1"}, Expected: map[string]any{"answer": "a"}},
		{Input: map[string]any{"q": "2"}, Expected: map[string]any{"answer": "a"}},
		{Input: map[string]any{"q": "3"}, Expected: map[string]any{"answer": "a"}},
		{Input: map[string]any{"q": "4"}, Expected: map[string]any{"answer": "a"}},
	}
	if err := setSeedTrainScores(&state, len(train), []float64{0, 0, 0, 0}); err != nil {
		t.Fatalf("setSeedTrainScores: %v", err)
	}
	if _, err := acceptCandidate(&state, len(train), acceptCandidateParams{
		ParentIDs: []int{0}, ProposalKind: proposalReflection, MutatedModule: "retriever",
		CreatedAtIter: 1, Prompts: Candidate{"retriever": "r1", "answer": "a0"}, TrainScores: []float64{0, 0, 0, 0},
	}); err != nil {
		t.Fatalf("accept c1: %v", err)
	}
	if _, err := acceptCandidate(&state, len(train), acceptCandidateParams{
		ParentIDs: []int{0}, ProposalKind: proposalReflection, MutatedModule: "answer",
		CreatedAtIter: 2, Prompts: Candidate{"retriever": "r0", "answer": "a1"}, TrainScores: []float64{0, 0, 0, 0},
	}); err != nil {
		t.Fatalf("accept c2: %v", err)
	}

	evaluator := &scriptedEvaluator{
		trainSize: len(train),
		scoreFor: func(c Candidate, examples []program.Example) float64 {
			if c["retriever"] == "r1" && c["answer"] == "a1" {
				return 1
			}
			return 0
		},
	}
	tracker := newMergeTracker()
	rng := rand.New(rand.NewSource(42))

	handled, accepted, err := tryMergeIteration(context.Background(), mergeIterationParams{
		state:         &state,
		writer:        newRunWriter("", false),
		tracker:       tracker,
		evaluator:     evaluator,
		train:         train,
		trainLen:      len(train),
		minibatchSize: 2,
		budget:        100,
		rng:           rng,
		iter:          2,
	})
	if err != nil {
		t.Fatalf("tryMergeIteration: %v", err)
	}
	if !handled || !accepted {
		t.Fatalf("handled=%v accepted=%v, want true/true", handled, accepted)
	}
	if len(state.Candidates) != 4 {
		t.Fatalf("candidates = %d, want 4 (seed + 2 reflections + merge)", len(state.Candidates))
	}
	merged := state.Candidates[3]
	if merged.ProposalKind != proposalMerge || len(merged.ParentIDs) != 2 {
		t.Fatalf("merged record = %#v", merged)
	}
	if merged.Prompts["retriever"] != "r1" || merged.Prompts["answer"] != "a1" {
		t.Fatalf("merged prompts = %#v, want r1/a1", merged.Prompts)
	}
}
