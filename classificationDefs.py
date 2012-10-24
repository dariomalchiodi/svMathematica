from cvxopt import matrix,solvers
from numpy import dot,array

def kronecker_delta(i,j):
	if i==j:
		return 1.0;
	else:
		return 0.0

def chop(data):
	tolerance=10**-4
	if data<tolerance:
		return 0.0
	else:
		return data
def chop_c(data,c):
	tolerance=10**-4
	if data<tolerance:
		return 0.0
	elif c-data<tolerance:
		return c
	else:
		return data

def svm_classification(patterns,labels,kernel):
	m=len(patterns)
	Q=matrix([[labels[i]*labels[j]*(kernel(patterns[i],patterns[j])) for i in range(m)] for j in range(m)])
	p=matrix([-1.0 for i in range(m)])
	g=[[-1.0*kronecker_delta(i,j) for i in range(m)] for j in range(m)]
	G=matrix(g)
	hl=[0.0 for i in range(m)]
	h=matrix(hl)
	A=matrix(labels,(1,m))
	b=matrix(0.0)
	sol=solvers.qp(Q,p,G,h,A,b)
	return map(lambda x:chop(x),list(sol['x']))


def svm_classification_c(patterns,labels,c,kernel):
	m=len(patterns)
	Q=matrix([[labels[i]*labels[j]*(kernel(patterns[i],patterns[j])) for i in range(m)] for j in range(m)])
	p=matrix([-1.0 for i in range(m)])
	g=[[-1.0*kronecker_delta(i,j) for i in range(m)]+[kronecker_delta(i,j) for i in range(m)] for j in range(m)]
	G=matrix(g)
	hl=[0.0 for i in range(m)]+[c for i in range(m)]
	h=matrix(hl)
	A=matrix(labels,(1,m))
	b=matrix(0.0)
	sol=solvers.qp(Q,p,G,h,A,b)
	return map(lambda x:chop_c(x,c),list(sol['x']))
