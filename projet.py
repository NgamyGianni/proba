import utils
import pandas as pd  # package for high-performance, easy-to-use data structures and data analysis
import numpy as np  # fundamental package for scientific computing with Python
import math
import scipy

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


def getPrior(data):
	som = 0

	for e in data["target"]:
		if e == 1:
			som += 1

	p = som / len(data.target)
	inf = p - 1.96 * math.sqrt(p * (1 - p) / len(data.target))
	sup = p + 1.96 * math.sqrt(p * (1 - p) / len(data.target))
	return {"estimation": p, "min5pourcent": inf, "max5pourcent": sup}


class APrioriClassifier(utils.AbstractClassifier):

	def __init__(self):
		pass

	def estimClass(self, attrs):
		return 1

	def statsOnDF(self, data):
		estim = self.estimClass(None)
		vp = 0
		vn = 0
		fp = 0
		fn = 0

		for e in data["target"]:
			if e == 0:
				fp += 1
			else:
				vp += 1

		return {"VP": vp, "VN": vn, "FP": fp, "FN": fn, "Précision": vp / (vp + fp), "Rappel": vp / (vp + fn)}


def P2D_l(df, attr):
	l = pd.crosstab(df.target, [df[attr]])
	tmp = {e for e in df[attr]}
	res = {0: {e: {} for e in df[attr]}, 1: {e: {} for e in df[attr]}}
	sm0 = 0
	sm1 = 0

	for i in tmp:
		sm0 += l[i][0]
		sm1 += l[i][1]

	d0 = res[0]
	d1 = res[1]

	for i in tmp:
		d0[i] = l[i][0] / sm0
		d1[i] = l[i][1] / sm1

	return res


def P2D_p(df, attr):
	l = pd.crosstab(df.target, [df[attr]])
	tmp = {e for e in df[attr]}
	res = {e: {} for e in df[attr]}

	for i in tmp:
		res[i] = {1: l[i][1] / (l[i][0] + l[i][1]), 0: l[i][0] / (l[i][0] + l[i][1])}
	return res

class ML2DClassifier(APrioriClassifier):

	def __init__(self, df, attr):
		self.df = df
		self.attr = attr
		self.P2Dl = P2D_l(df, attr)

	def estimClass(self, e):
		if self.P2Dl[0][e[self.attr]] >= self.P2Dl[1][e[self.attr]]:
			return 0
		return 1

	def statsOnDF(self, data):
		vp = 0
		vn = 0
		fp = 0
		fn = 0

		for i in range(len(data)):
			e = data["target"][i]
			if e == 0:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					fp += 1
				else:
					vn += 1
			else:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					vp += 1
				else:
					fn += 1

		return {"VP": vp, "VN": vn, "FP": fp, "FN": fn, "Précision": vp / (vp + fp), "Rappel": vp / (vp + fn)}

class MAP2DClassifier(APrioriClassifier):

	def __init__(self, df, attr):
		self.df = df
		self.attr = attr
		self.P2Dp = P2D_p(df, attr)

	def estimClass(self, e):
		if self.P2Dp[e[self.attr]][0] >= self.P2Dp[e[self.attr]][1]:
			return 0
		return 1

	def statsOnDF(self, data):
		vp = 0
		vn = 0
		fp = 0
		fn = 0

		for i in range(len(data)):
			e = data["target"][i]
			if e == 0:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					fp += 1
				else:
					vn += 1
			else:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					vp += 1
				else:
					fn += 1

		return {"VP": vp, "VN": vn, "FP": fp, "FN": fn, "Précision": vp / (vp + fp), "Rappel": vp / (vp + fn)}

def nbParams(df, L):
	res = 1

	for e in L:
		res *= len({e for e in df[e]})

	return res * 8

def nbParamsIndep(df):
	res = 0

	for e in df:
		res += len({e for e in df[e]})

	return res * 8

def drawNaiveBayes(df, attr):
	res = ""

	for a in {e for e in df}-{attr}:
		res += attr+"->"+a+";"

	res = res[:len(res)-1]
	return utils.drawGraph(res)

def nbParamsNaiveBayes(df, attr, attrs):
	res = 0
	tmp = len({e for e in df[attr]})

	for e in attrs:
		if e != attr:
			res += len({e for e in df[e]}) * tmp

	res += tmp
	return res * 8

def proba(data, attr, df):
	l = pd.crosstab(df.target, [df[attr]])
	res = 0
	
	for e in l:
		res += l[e][0]+l[e][1]

	return (l[data][0]+l[data][1])/res

class MLNaiveBayesClassifier(APrioriClassifier):

	def __init__(self, df):
		self.df = df
		self.p = {e : P2D_l(df, e) for e in df if e != "target"}

	def estimProbas(self, e):
		res = {0:1, 1:1}
		for key in e:
			if key != "target":
				if e[key] in self.p[key][0]:
					res[0] *= self.p[key][0][e[key]]
					res[1] *= self.p[key][1][e[key]]
				else:
					res[0] *= 0
					res[1] *= 0
		return res

	def estimClass(self, e):
		if self.estimProbas(e)[0] >= self.estimProbas(e)[1]:
			return 0
		return 1

	def statsOnDF(self, data):
		vp = 0
		vn = 0
		fp = 0
		fn = 0

		for i in range(len(data)):
			e = data["target"][i]
			if e == 0:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					fp += 1
				else:
					vn += 1
			else:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					vp += 1
				else:
					fn += 1

		return {"VP": vp, "VN": vn, "FP": fp, "FN": fn, "Précision": vp / (vp + fp), "Rappel": vp / (vp + fn)}

class MAPNaiveBayesClassifier(APrioriClassifier):

	def __init__(self, df):
		self.df = df
		self.p = {e: P2D_l(df, e) for e in df if e != "target"}
		self.p0 = proba(0, "target", self.df)
		self.p1 = proba(1, "target", self.df)

	def estimProbas(self, e):
		res = {0: 1, 1: 1}

		total = 1

		for key in e:
			if key != "target":
				if e[key] in self.p[key][0]:
					res[0] *= self.p[key][0][e[key]]
					res[1] *= self.p[key][1][e[key]]
					total *= (self.p[key][0][e[key]]*self.p0)+(self.p[key][1][e[key]]*self.p1)
				else:
					res[0] *= 0
					res[1] *= 0
		res[0] = (self.p0*res[0])/total
		res[1] = (self.p1*res[1])/total
		total2 = res[0]+res[1]
		res[0] = res[0]/total2
		res[1] = res[1]/total2
		return res

	def estimClass(self, e):
		if self.estimProbas(e)[0] >= self.estimProbas(e)[1]:
			return 0
		return 1

	def statsOnDF(self, data):
		vp = 0
		vn = 0
		fp = 0
		fn = 0

		for i in range(len(data)):
			e = data["target"][i]
			if e == 0:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					fp += 1
				else:
					vn += 1
			else:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					vp += 1
				else:
					fn += 1

		return {"VP": vp, "VN": vn, "FP": fp, "FN": fn, "Précision": vp / (vp + fp), "Rappel": vp / (vp + fn)}

def isIndepFromTarget(df, attr, x):
	l = pd.crosstab(df.target, [df[attr]])
	return scipy.stats.chi2_contingency(l)[1] >= x

class ReducedMLNaiveBayesClassifier(APrioriClassifier):
	def __init__(self, df, x):
		self.df = df
		self.attrs = [attr for attr in df if not(isIndepFromTarget(df, attr, x)) and attr != "target"]
		self.p = {e: P2D_l(df, e) for e in self.attrs} 

	def estimProbas(self, e):
		res = {0:1, 1:1}
		for key in self.attrs:
			if e[key] in self.p[key][0]:
				res[0] *= self.p[key][0][e[key]]
				res[1] *= self.p[key][1][e[key]]
			else:
				res[0] *= 0
				res[1] *= 0
		return res

	def estimClass(self, e):
		if self.estimProbas(e)[0] >= self.estimProbas(e)[1]:
			return 0
		return 1

	def draw(self):
		return drawNaiveBayes(self.attrs, "target")

	def statsOnDF(self, data):
		vp = 0
		vn = 0
		fp = 0
		fn = 0

		for i in range(len(data)):
			e = data["target"][i]
			if e == 0:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					fp += 1
				else:
					vn += 1
			else:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					vp += 1
				else:
					fn += 1

		return {"VP": vp, "VN": vn, "FP": fp, "FN": fn, "Précision": vp / (vp + fp), "Rappel": vp / (vp + fn)}

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):

	def __init__(self, df, x):
		self.df = df
		self.attrs = [attr for attr in df if not(isIndepFromTarget(df, attr, x)) and attr != "target"]
		self.p = {e: P2D_l(df, e) for e in self.attrs}
		self.p0 = proba(0, "target", self.df)
		self.p1 = proba(1, "target", self.df)

	def estimProbas(self, e):
		res = {0: 1, 1: 1}

		total = 1

		for key in self.attrs:
			if e[key] in self.p[key][0]:
				res[0] *= self.p[key][0][e[key]]
				res[1] *= self.p[key][1][e[key]]
				total *= (self.p[key][0][e[key]]*self.p0)+(self.p[key][1][e[key]]*self.p1)
			else:
				res[0] *= 0
				res[1] *= 0
		res[0] = (self.p0*res[0])/total
		res[1] = (self.p1*res[1])/total
		total2 = res[0]+res[1]
		res[0] /= total2
		res[1] /= total2
		return res

	def estimClass(self, e):
		if self.estimProbas(e)[0] >= self.estimProbas(e)[1]:
			return 0
		return 1

	def draw(self):
		return drawNaiveBayes(self.attrs, "target")

	def statsOnDF(self, data):
		vp = 0
		vn = 0
		fp = 0
		fn = 0

		for i in range(len(data)):
			e = data["target"][i]
			if e == 0:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					fp += 1
				else:
					vn += 1
			else:
				if self.estimClass(utils.getNthDict(data, i)) == 1:
					vp += 1
				else:
					fn += 1

		return {"VP": vp, "VN": vn, "FP": fp, "FN": fn, "Précision": vp / (vp + fp), "Rappel": vp / (vp + fn)}

cl=MAPNaiveBayesClassifier(train)
for i in [0, 1, 2]:
	print("Estimation de la proba de l'individu {} par MAPNaiveBayesClassifier : {}".format(i,cl.estimProbas(utils.getNthDict(train,i))))
	print("Estimation de la classe de l'individu {} par MAPNaiveBayesClassifier : {}".format(i,cl.estimClass(utils.getNthDict(train,i)))) 
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))


#print(scipy.stats.chi2_contingency(pd.crosstab(train.target, [train["fbs"]])))
