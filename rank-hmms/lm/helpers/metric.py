import math

def perplexity(x, n):
    
    return math.exp(min(-x / n, 700))
    
class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 1e9
        
class LikelihoodMetric(Metric):
    def __init__(self, eps=1e-8):
        super(Metric, self).__init__()
        self.eps = eps
        self.tokens = 0.0
        self.elbo_value = 0.0
        self.total_likelihood = 0.0

    @property
    def score(self):
        return self.perplexity

    def __call__(self, likelihood, n_tokens, elbo = None):

        self.total_likelihood += likelihood.detach()
        self.tokens += n_tokens
        
        if elbo is not None:
            self.elbo_value += elbo.detach()

    @property
    def evidence(self):
        return self.total_likelihood
    
    @property
    def avg_likelihood(self):
        return self.total_likelihood / self.tokens

    @property
    def perplexity(self, max_num=700):
        return math.exp(min(-self.total_likelihood / self.tokens, max_num))
        
    @property
    def elbo(self):
        return self.elbo_value / self.tokens

    def __repr__(self):
        return "avg likelihood: {}, perp. :{}".format(self.avg_likelihood, self.perplexity)