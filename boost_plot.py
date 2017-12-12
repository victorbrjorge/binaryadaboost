#!/usr/bin/python
#-*- coding: utf-8 -*-
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np

def get_data():
	train_error = np.loadtxt('train_error')
	stump_error = np.loadtxt('stump_error')
	test_error = np.loadtxt('test_error')
	print test_error
	return train_error, stump_error, test_error

def plot_train_error():
	train_error, stump_error,_ = get_data()
	fig, ax = plt.subplots()

	it = range(1,301)
	ax.plot(it,train_error,'-', linewidth = 2, label=u'Erro do modelo atual', color='red')
	ax.plot(it,stump_error,'--', linewidth = 2, label=u'Erro do stump selecionado', color='blue')

	ax.ticklabel_format(useOffset=False, style='plain')

	ax.set_ylim([0,0.5])
	ax.set_xlim([0,310])
	ax.xaxis.grid()

	ax.legend()

	plt.title(u"Adaboost - validação cruzada com 5 conjuntos")

	plt.ylabel(u"Erro médio")
	plt.xlabel(u"Iteração")

	plt.show()

def plot_test_error():
	_, _, test_error = get_data()
	fig, ax = plt.subplots()

	it = range(50,301,50)
	ax.plot(it,test_error,'-', linewidth = 2, label=u'Erro no conjunto de teste', color='red')

	ax.ticklabel_format(useOffset=False, style='plain')

	ax.set_ylim([0,0.2])
	ax.xaxis.grid()

	ax.legend()

	plt.title(u"Adaboost - validação cruzada com 5 conjuntos")

	plt.ylabel(u"Erro médio")
	plt.xlabel(u"Numero de iterações para gerar o modelo")

	plt.show()

if __name__ == '__main__':
	plot_train_error()
	#plot_test_error()