package gepa

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/anath2/gepa-go/internal/program"
)

func TestNewRNGIsDeterministic(t *testing.T) {
	left := newRNG(42)
	right := newRNG(42)
	var leftSeq, rightSeq []int
	for range 5 {
		leftSeq = append(leftSeq, left.Intn(1000))
		rightSeq = append(rightSeq, right.Intn(1000))
	}
	if !reflect.DeepEqual(leftSeq, rightSeq) {
		t.Fatalf("rng sequences differ: left=%v right=%v", leftSeq, rightSeq)
	}
}

func TestSampleMinibatchDeterministicSubset(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	got, err := sampleMinibatch(rng, 6, 3)
	if err != nil {
		t.Fatalf("sampleMinibatch() unexpected error: %v", err)
	}
	want := []int{0, 2, 5}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sampleMinibatch() = %v, want %v", got, want)
	}
}

func TestSampleMinibatchReturnsAllIndicesWhenSizeCoversDataset(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	got, err := sampleMinibatch(rng, 3, 5)
	if err != nil {
		t.Fatalf("sampleMinibatch() unexpected error: %v", err)
	}
	want := []int{0, 1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sampleMinibatch() = %v, want %v", got, want)
	}
}

func TestSampleMinibatchRejectsInvalidInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	if _, err := sampleMinibatch(rng, 0, 3); err == nil {
		t.Fatal("sampleMinibatch(total=0) error = nil, want error")
	}
	if _, err := sampleMinibatch(rng, 5, 0); err == nil {
		t.Fatal("sampleMinibatch(size=0) error = nil, want error")
	}
}

func TestMeanScore(t *testing.T) {
	got, err := meanScore([]float64{0, 1, 0.5})
	if err != nil {
		t.Fatalf("meanScore() unexpected error: %v", err)
	}
	if got != 0.5 {
		t.Fatalf("meanScore() = %v, want 0.5", got)
	}
	if _, err := meanScore(nil); err == nil {
		t.Fatal("meanScore(nil) error = nil, want error")
	}
}

func TestStrictlyImproves(t *testing.T) {
	if !strictlyImproves(0.2, 0.5) {
		t.Fatal("strictlyImproves(0.2, 0.5) = false, want true")
	}
	if strictlyImproves(0.5, 0.5) {
		t.Fatal("strictlyImproves(0.5, 0.5) = true, want false")
	}
	if strictlyImproves(0.6, 0.4) {
		t.Fatal("strictlyImproves(0.6, 0.4) = true, want false")
	}
}

func TestHasBudget(t *testing.T) {
	if !hasBudget(0, 10, 3) {
		t.Fatal("hasBudget(0,10,3) = false, want true")
	}
	if hasBudget(8, 10, 3) {
		t.Fatal("hasBudget(8,10,3) = true, want false")
	}
}

func TestModuleNameAtIterationRoundRobin(t *testing.T) {
	prog := program.Program{
		Modules: []program.Module{
			{Name: "a", Prompt: "a"},
			{Name: "b", Prompt: "b"},
			{Name: "c", Prompt: "c"},
		},
	}
	for iter, want := range []string{"a", "b", "c", "a"} {
		got, err := moduleNameAtIteration(prog, iter)
		if err != nil {
			t.Fatalf("moduleNameAtIteration(%d) unexpected error: %v", iter, err)
		}
		if got != want {
			t.Fatalf("moduleNameAtIteration(%d) = %q, want %q", iter, got, want)
		}
	}
}
