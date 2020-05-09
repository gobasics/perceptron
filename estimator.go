package perceptron

type Estimator func([]float64) float64

func (fn Estimator) Estimate(x []float64) float64 {
	return fn(x)
}
