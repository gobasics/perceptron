package perceptron

import (
	"math/rand"
	"time"
)

// Perceptron implements the perceptron algorithm.
type Perceptron []float64

// Estimator returns a closure (referencing activate function) that takes a set of
// examples and returns a prediction
func (p Perceptron) Estimator(activate func(float64) float64) Estimator {
	return func(x []float64) float64 {
		y := p[0]
		for k := 0; k < len(x); k++ {
			y += p[k+1] * x[k]
		}
		return activate(y)
	}
}

// init initializes weights in [-1, 1]
func (p Perceptron) init() {
	rand.Seed(time.Now().UnixNano())
	for k := range p {
		p[k] = rand.Float64() - float64(rand.Intn(1))
	}
}

// Minimizer returns a closure (referencing activation function, cost functions and learning rate)
// that takes a training example, reduces the error and returns the pre-reduce prediction.
func (p Perceptron) Minimizer(activate func(float64) float64, cost func(float64) float64, r float64) Minimizer {
	estimator := p.Estimator(activate)
	return func(x []float64, yTrue float64) float64 {
		yEstimate := estimator.Estimate(x)
		delta := cost(yTrue-yEstimate) * r
		p[0] += delta
		for k := range x {
			p[k+1] += x[k] * delta
		}
		return yEstimate
	}
}

// New creates and initializes a Perceptron of size parameters
func New(size int) Perceptron {
	size++
	p := make(Perceptron, size)
	p.init()
	return p
}
