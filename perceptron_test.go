package perceptron

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

func TestPerceptron(t *testing.T) {
	f, err := os.Open("testdata/examples.golden")
	if err != nil {
		t.Fatalf("want nil, got %v", err)
	}

	type Example struct {
		X []float64 `json:"X"`
		Y float64   `json:"Y"`
	}

	var examples []Example

	if err = json.NewDecoder(f).Decode(&examples); err != nil {
		t.Fatalf("want nil, got %v", err)
	}

	var activate = func(z float64) float64 {
		if 0. > z {
			return 0.
		}
		return 1.
	}

	var square = func(y float64) float64 {
		var d float64 = 1
		if 0. > y {
			d = -1.
		}
		return y * y * d
	}

	var equal = func(x, y float64) bool {
		return math.Abs(x-y) <= 1e-9
	}

	p := New(2)

	const LR = 0.01
	minimizer := p.Minimizer(activate, square, LR)
	estimator := p.Estimator(activate)

	type Accuracy float64
	for k := 0; k < 10; k++ {
		t.Run(strconv.Itoa(k), func(t *testing.T) {
			var after, before, total Accuracy
			var minimized int
			for j := range examples {
				yTrue := examples[j].Y
				yEstimateBefore := minimizer.Minimize(examples[j].X, yTrue)
				total++
				if equal(yTrue, yEstimateBefore) {
					before++
				}

				yEstimateAfter := estimator.Estimate(examples[j].X)
				if equal(yTrue, yEstimateAfter) {
					after++
				}

				if after > before {
					minimized++
				}
			}

			epoch := k + 1

			t.Logf("epoch: %d, accurate: %.2f, minimized: %d of %d training steps, LR: %.2f\n", epoch, after/total, minimized, len(examples), LR)
			const want Accuracy = .5
			got := after / total
			if want > got {
				t.Errorf("want %.2f, got %.2f", want, got)
			}
		})
	}
}
