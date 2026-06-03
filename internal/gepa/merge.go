package gepa

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/anath2/gepa-go/internal/program"
)

// mergePair is a pair of candidate IDs and an ancestor ID.
type mergePair struct {
	ID1      int
	ID2      int
	Ancestor int
}

// mergeTracker tracks which merge triplets have already been attempted.
type mergeTracker struct {
	attempted map[mergePair]struct{}
}

// mergeIterationParams is the context for a single merge iteration.
type mergeIterationParams struct {
	state         *poolState
	writer        runWriter
	tracker       *mergeTracker
	evaluator     Evaluator
	train         []program.Example
	trainLen      int
	minibatchSize int
	budget        int
	rng           *rand.Rand
	iter          int
}

func newMergeTracker() *mergeTracker {
	return &mergeTracker{
		attempted: map[mergePair]struct{}{},
	}
}

func canonicalMergePair(id1, id2, ancestor int) mergePair {
	if id1 > id2 {
		id1, id2 = id2, id1
	}
	return mergePair{ID1: id1, ID2: id2, Ancestor: ancestor}
}

func (t *mergeTracker) recordTriplet(id1, id2, ancestor int) {
	t.attempted[canonicalMergePair(id1, id2, ancestor)] = struct{}{}
}

func (t *mergeTracker) alreadyAttempted(id1, id2, ancestor int) bool {
	_, ok := t.attempted[canonicalMergePair(id1, id2, ancestor)]
	return ok
}

// module matches the ancestor in exactly one descendant and differs in the other.
func mergeDesirable(ancestor, i, j Candidate) bool {
	for name := range ancestor {
		pa := ancestor[name]
		pi := i[name]
		pj := j[name]
		if (pa == pi && pj != pi) || (pa == pj && pi != pj) {
			return true
		}
	}
	return false
}

func candidateMeanScore(state poolState, id int) (float64, error) {
	if id < 0 || id >= len(state.TrainScores) {
		return 0, fmt.Errorf("candidate %d out of range", id)
	}
	return meanScore(state.TrainScores[id])
}

func filterMergeAncestors(
	tracker *mergeTracker,
	id1, id2 int,
	common []int,
	scores []float64,
	candidates []candidateRecord,
) []int {
	var filtered []int
	for _, ancestor := range common {
		if tracker.alreadyAttempted(id1, id2, ancestor) {
			continue
		}
		if scores[ancestor] > scores[id1] || scores[ancestor] > scores[id2] {
			continue
		}
		if !mergeDesirable(candidates[ancestor].Prompts, candidates[id1].Prompts, candidates[id2].Prompts) {
			continue
		}
		filtered = append(filtered, ancestor)
	}
	return filtered
}

// Recursively walks the candidate tree to find all ancestors.
func candidateAncestors(state poolState, id int) []int {
	seen := map[int]struct{}{}
	var walk func(int)
	walk = func(node int) {
		if node < 0 || node >= len(state.Candidates) {
			return
		}
		for _, parent := range state.Candidates[node].ParentIDs {
			if _, ok := seen[parent]; ok {
				continue
			}
			seen[parent] = struct{}{}
			walk(parent)
		}
	}
	walk(id)
	out := make([]int, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	return out
}

// commonAncestors finds all ancestors that are common to both candidates.
func commonAncestors(state poolState, id1, id2 int) []int {
	a1 := candidateAncestors(state, id1)
	set := map[int]struct{}{}
	for _, id := range a1 {
		set[id] = struct{}{}
	}
	var common []int
	for _, id := range candidateAncestors(state, id2) {
		if _, ok := set[id]; ok {
			common = append(common, id)
		}
	}
	return common
}

// findMergePair finds a merge pair from the given candidates.
func findMergePair(state poolState, tracker *mergeTracker, rng *rand.Rand, mergeCandidates []int) (mergePair, bool) {
	if len(mergeCandidates) < 2 || len(state.Candidates) < 3 {
		return mergePair{}, false
	}

	scores := make([]float64, len(state.Candidates))
	for i := range state.Candidates {
		mean, err := candidateMeanScore(state, i)
		if err != nil {
			return mergePair{}, false
		}
		scores[i] = mean
	}

	ancestorCache := map[int][]int{}
	ancestorSetCache := map[int]map[int]struct{}{}
	ancestorsFor := func(id int) []int {
		if ancestors, ok := ancestorCache[id]; ok {
			return ancestors
		}
		ancestors := candidateAncestors(state, id)
		ancestorCache[id] = ancestors
		return ancestors
	}
	ancestorSetFor := func(id int) map[int]struct{} {
		if set, ok := ancestorSetCache[id]; ok {
			return set
		}
		set := map[int]struct{}{}
		for _, ancestor := range ancestorsFor(id) {
			set[ancestor] = struct{}{}
		}
		ancestorSetCache[id] = set
		return set
	}

	var viablePairs []mergePair
	for left := 0; left < len(mergeCandidates)-1; left++ {
		for right := left + 1; right < len(mergeCandidates); right++ {
			i := mergeCandidates[left]
			j := mergeCandidates[right]
			if i == j {
				continue
			}
			if i > j {
				i, j = j, i
			}

			ancestorsI := ancestorsFor(i)
			ancestorsJ := ancestorsFor(j)
			if containsInt(ancestorsI, j) || containsInt(ancestorsJ, i) {
				continue
			}

			ancestorSetI := ancestorSetFor(i)
			var common []int
			for _, ancestor := range ancestorsJ {
				if _, ok := ancestorSetI[ancestor]; ok {
					common = append(common, ancestor)
				}
			}

			viable := filterMergeAncestors(tracker, i, j, common, scores, state.Candidates)
			for _, ancestor := range viable {
				viablePairs = append(viablePairs, mergePair{ID1: i, ID2: j, Ancestor: ancestor})
			}
		}
	}

	if len(viablePairs) == 0 {
		return mergePair{}, false
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(0))
	}
	return viablePairs[rng.Intn(len(viablePairs))], true
}

func containsInt(list []int, target int) bool {
	for _, v := range list {
		if v == target {
			return true
		}
	}
	return false
}

// common ancestor prompts and take complementary module prompts from descendants.
func combineMergedCandidate(state poolState, pair mergePair, rng *rand.Rand) Candidate {
	ancestor := state.Candidates[pair.Ancestor].Prompts
	id1 := state.Candidates[pair.ID1].Prompts
	id2 := state.Candidates[pair.ID2].Prompts

	merged := cloneCandidate(ancestor)
	score1, _ := candidateMeanScore(state, pair.ID1)
	score2, _ := candidateMeanScore(state, pair.ID2)

	for moduleName := range merged {
		pa := ancestor[moduleName]
		pi := id1[moduleName]
		pj := id2[moduleName]
		switch {
		// ancestor and id1, same prompt, different from id2
		case pa == pi && pj != pi:
			merged[moduleName] = pj
		// ancestor and id2, same prompt, different from id1
		case pa == pj && pi != pj:
			merged[moduleName] = pi
		// ancestor and id1, different from id2, different from ancestor
		case pi != pj && pa != pi && pa != pj:
			if score1 > score2 {
				merged[moduleName] = pi
			} else if score2 > score1 {
				merged[moduleName] = pj
				// ancestor and id1, different from id2, different from ancestor, random choice
			} else if rng.Intn(2) == 0 {
				merged[moduleName] = pi
			} else {
				merged[moduleName] = pj
			}
		default:
			merged[moduleName] = pi
		}
	}
	return merged
}

// minibatchMeanFromScores calculates the mean score of a minibatch.
func minibatchMeanFromScores(scores []float64, batchIndices []int) (float64, error) {
	values := make([]float64, 0, len(batchIndices))
	for _, idx := range batchIndices {
		if idx < 0 || idx >= len(scores) {
			return 0, fmt.Errorf("minibatch mean: index %d out of range", idx)
		}
		values = append(values, scores[idx])
	}
	return meanScore(values)
}

// mergeAccepts reports whether a proposal beats its best parent on a minibatch mean score.
func mergeAccepts(proposalMean, bestParentMean float64) bool {
	return strictlyImproves(bestParentMean, proposalMean)
}

// tryMergeIteration attempts one System-Aware Merge step (paper Algs. 3 & 4).
// Returns handled=true when merge ran (accepted or rejected); false when no
// compatible pair was found and the caller should fall through to reflection.
// When handled is true, accepted reports whether the merged candidate entered the pool.
func tryMergeIteration(ctx context.Context, p mergeIterationParams) (handled bool, accepted bool, err error) {
	frontier, err := paretoFrontier(*p.state)
	if err != nil {
		return false, false, err
	}
	if len(frontier) < 2 {
		return false, false, nil
	}

	pair, ok := findMergePair(*p.state, p.tracker, p.rng, frontier)
	if !ok {
		return false, false, nil
	}

	parentIDs := []int{pair.ID1, pair.ID2}
	batchIndices, err := sampleMinibatch(p.rng, p.trainLen, p.minibatchSize)
	if err != nil {
		return false, false, err
	}
	batch := examplesAtIndices(p.train, batchIndices)

	parent1Mean, err := minibatchMeanFromScores(p.state.TrainScores[pair.ID1], batchIndices)
	if err != nil {
		return false, false, err
	}
	parent2Mean, err := minibatchMeanFromScores(p.state.TrainScores[pair.ID2], batchIndices)
	if err != nil {
		return false, false, err
	}
	bestParentMean := parent1Mean
	if parent2Mean > bestParentMean {
		bestParentMean = parent2Mean
	}

	if err := p.writer.mergeRequested(*p.state, parentIDs, pair.Ancestor, batchIndices); err != nil {
		return false, false, err
	}

	merged := combineMergedCandidate(*p.state, pair, p.rng)
	p.tracker.recordTriplet(pair.ID1, pair.ID2, pair.Ancestor)

	proposalEval, err := evaluateCandidate(ctx, p.state, p.evaluator, merged, batch)
	if err != nil {
		return false, false, err
	}

	if err := p.writer.mergeEvaluated(*p.state, parentIDs, batchIndices, bestParentMean, proposalEval.Mean); err != nil {
		return false, false, err
	}

	if !mergeAccepts(proposalEval.Mean, bestParentMean) {
		if err := p.writer.mergeRejected(*p.state, parentIDs, batchIndices, bestParentMean, proposalEval.Mean); err != nil {
			return false, false, err
		}
		return true, false, nil
	}

	if !hasBudget(p.state.MetricCalls, p.budget, p.trainLen) {
		return true, false, nil
	}

	fullEval, err := evaluateCandidate(ctx, p.state, p.evaluator, merged, p.train)
	if err != nil {
		return false, false, err
	}

	newID, err := acceptCandidate(p.state, p.trainLen, acceptCandidateParams{
		ParentIDs:     parentIDs,
		ProposalKind:  proposalMerge,
		CreatedAtIter: p.iter + 1,
		Prompts:       merged,
		TrainScores:   fullEval.Scores,
	})
	if err != nil {
		return false, false, err
	}
	if err := p.writer.persistAcceptedMerge(*p.state, newID, bestParentMean, proposalEval.Mean, parentIDs, batchIndices); err != nil {
		return false, false, err
	}
	return true, true, nil
}
