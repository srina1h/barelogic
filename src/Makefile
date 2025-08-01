# Default is show help; e.g.
#
#    make 
#
# prints the help text.

PYTHON_VER = python3.10

SHELL     := bash
MAKEFLAGS += --warn-undefined-variables
.SILENT:

LOUD = \033[1;34m#
HIGH = \033[1;33m#
SOFT = \033[0m#

Top=$(shell git rev-parse --show-toplevel)
Tmp  ?= $(HOME)/tmp

help      :  ## show help
	gawk -f $(Top)/etc/help.awk $(MAKEFILE_LIST) 

pull: ## update from main
	git pull

push: ## commit to main
	- echo -en "$(LOUD)Why this push? $(SOFT)" 
	- read x ; git commit -am "$$x" ;  git push
	- git status

sh: ## run my shell
	bash --init-file  $(Top)/etc/dotshellrc -i

luac:
	luacheck bll.lua --ignore 61*
lint: ## lint all python in this directory
	export PYTHONPATH="..:$$PYTHONPATH"; \
	pylint --disable=W0311,C0303,C0116,C0321,C0103 \
		    --disable=C0410,C0115,C3001,R0903,E1101 \
		    --disable=E701,W0108,W0106,W0718,W0201   bl.py

../docs/%.html : %.py
	pycco -d $(Top)/docs  $^
	echo "pre { font-size: small;} p { text-align:right; }" >> $(Top)/docs/pycco.css
	gawk '/<h1>/ {print "<div class=docs>";                       \
                while(getline x < "../etc/head.html") {print x}; \
                print "<h1>'$^'</h1></div>";                  \
                next} 1' $@ > tmp.tmp
	mv tmp.tmp $@

../docs/%.html : %.lua
	pycco -d $(Top)/docs  $^
	echo "pre { font-size: small;} p { text-align:right; }" >> $(Top)/docs/pycco.css
	gawk '/<h1>/ {print "<div class=docs>";                       \
                while(getline x < "../etc/head.html") {print x}; \
                print "<h1>'$^'</h1></div>";                  \
                next} 1' $@ > tmp.tmp
	mv tmp.tmp $@

~/tmp/%.pdf: %.py  ## make doco: .py ==> .pdf
	mkdir -p ~/tmp
	echo "pdf-ing $@ ... "
	a2ps                 \
		-Br                 \
		--chars-per-line=90 \
		--file-align=fill      \
		--line-numbers=1        \
		--pro=color               \
		--left-title=""            \
		--borders=no             \
	    --left-footer="$<  "               \
	    --right-footer="page %s. of %s#"               \
		--columns 3                 \
		-M letter                     \
	  -o	 $@.ps $<
	ps2pdf $@.ps $@; rm $@.ps
	open $@

~/tmp/%.pdf : %.lua  Makefile
	@echo "pdfing : $@ ... "
	@a2ps -Bj --landscape                           \
		--chars-per-line=90 \
		--line-numbers=1                    \
		--highlight-level=normal  \
		--columns 3                 \
		--borders=no --pro=color \
		--right-footer="" --left-footer=""    \
		--pretty-print=../etc/lua.ssh             \
		--footer="page %p."                     \
		-M letter -o $@.ps $<
	@ps2pdf $@.ps $@; rm $@.ps
	open $@

actsReport:
	cat ~/tmp/acts_*.out | sort | uniq | sed "s/[{}']//g" | sort -n -k 2  | column -s, -t | grep --color a

acts: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d, $^, (python3 -B bl.py --acts $d | tee ~/tmp/$@_$(notdir $d).out &); )

trees: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d, $^, python3 -B bl.py --tree $d;)

rules: ../../moot/optimize/[bchmp]*/*.csv
	$(foreach d, $^, python3 -B bl.py -l 5 --rules $d | column -t;)

after: ../../moot/optimize/[bchmp]*/*.csv
	mkdir -p ~/tmp
	{ $(foreach d, $^, (python3 -B bl.py -l 2  --after $d &);) } | tee  ~/tmp/$@.out
	bash after.sh ~/tmp/$@.out

afterDumb: ../../moot/optimize/[bchmp]*/*.csv
	mkdir -p ~/tmp
	{ $(foreach d, $^, (python3 -B bl.py -l 2  --afterDumb $d &);) } | tee  ~/tmp/$@.out
	bash after.sh ~/tmp/$@.out

stats: ../../moot/optimize/[bchmp]*/*.csv
	echo "     x      y   rows"
	echo "------ ------ ------"
	{ $(foreach d, $^, gawk -F, -f stats.awk $d;) } | sort -n

#l=2
# S   10 30 50  70 90 <== percentile
# === == == ==  == ==
# 200 21 78 93  98 100
# 100 72 89 97 100 100
#  50 66 93 97 100 100
#  40 65 85 97 100 100
#  30 65 83 93 100 100
#  20 70 89 97 100 100

# ▶ py bl.py -l 2 -S 30 --after ../../moot/optimize/process/nasa93dem.csv
# o{:win 97 :samples 30 :mu1 0.54 :mu2 0.525 :lo1 0.387 :lo2 0.391 :file "nasa93dem.csv"}


# py bl.py -l 2 --tree ../../moot/optimize/process/nasa93dem.csv
# o{:mu1 0.54 :mu2 0.437 :sd1 0.051 :sd2 0.019}
# ▶ py bl.py -l 2 -S 30 --tree ../../moot/optimize/process/nasa93dem.csv
# o{:mu1 0.54 :mu2 0.46 :sd1 0.051 :sd2 0.043}
# [30] 0.531
# |  rely == l [2] 0.438
# |  rely == h [10] 0.52
# |  |  data == h [2] 0.481
# |  |  data == n [6] 0.532
# |  |  |  pmat == l [3] 0.504
# |  |  |  |  cplx == vh [2] 0.506
# |  |  |  pmat == n [2] 0.569
# |  rely == n [14] 0.536
# |  |  data == n [5] 0.521
# |  |  |  pmat == l [2] 0.527
# |  |  |  pmat == n [2] 0.54
# |  |  data == l [7] 0.541
# |  |  |  pcap == h [5] 0.54
# |  |  |  |  cplx == n [3] 0.526
# |  |  |  |  cplx == h [2] 0.562
# |  |  |  pcap == vh [2] 0.543
# |  rely == vh [4] 0.587
# |  |  pmat == h [2] 0.561
# |  |  pmat == l [2] 0.613

# Run eg__nbfew on all preprocessed datasets in bl mode
parent_dir = $(shell pwd)/..
nbfew_bl: 
	@echo "Running eg__nbfew on all preprocessed datasets (bl mode)"
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/preprocessed_Hall_num.csv
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/preprocessed_Kitchenham_num.csv
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/preprocessed_Wahono_num.csv

# Run eg__nbfew on all preprocessed datasets in sklearn mode
nbfew_sk: 
	@echo "Running eg__nbfew on all preprocessed datasets (sklearn mode)"
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/preprocessed_Hall_num.csv
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/preprocessed_Kitchenham_num.csv
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/preprocessed_Wahono_num.csv

nbfew_bl_min:
	@echo "Running eg__nbfew on all minimally preprocessed datasets (bl mode)"
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/minimally_processed_Hall_num.csv
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/minimally_processed_Kitchenham_num.csv
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv
	python3 -B bl-added.py -M bl --nbfew $(parent_dir)/test_data/minimally_processed_Wahono_num.csv

nbfew_sk_min:
	@echo "Running eg__nbfew on all minimally preprocessed datasets (sklearn mode)"
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/minimally_processed_Hall_num.csv
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/minimally_processed_Kitchenham_num.csv
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv
	python3 -B bl-added.py -M sklearn --nbfew $(parent_dir)/test_data/minimally_processed_Wahono_num.csv

nbal_bl:
	@echo "Running eg__nbAL on all preprocessed datasets (bl mode)"
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Hall_num.csv
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Kitchenham_num.csv
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Wahono_num.csv

nbal_sk:
	@echo "Running eg__nbAL on all preprocessed datasets (sklearn mode)"
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Hall_num.csv
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Kitchenham_num.csv
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Wahono_num.csv

nbal_bl_min:
	@echo "Running eg__nbAL on all minimally preprocessed datasets (bl mode)"
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num.csv
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Kitchenham_num.csv
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv
	python3.10 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Wahono_num.csv

nbal_sk_min:
	@echo "Running eg__nbAL on all minimally preprocessed datasets (sklearn mode)"
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num.csv
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Kitchenham_num.csv
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv
	python3.10 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Wahono_num.csv

exp_2_nbal_hall:
	@echo "Running eg__nbAL on Hall sk + bl / minimal + preprocessed"

	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Hall_num.csv
	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num.csv

	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Hall_num.csv
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num.csv

exp_2_nbal_kitchenham:
	@echo "Running eg__nbAL on Kitchenham sk + bl / minimal + preprocessed"
	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Kitchenham_num.csv
	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Kitchenham_num.csv

	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Kitchenham_num.csv
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Kitchenham_num.csv

exp_2_nbal_Wahono:
	@echo "Running eg__nbAL on Wahono sk + bl / minimal + preprocessed"
	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Wahono_num.csv
	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Wahono_num.csv

	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Wahono_num.csv
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Wahono_num.csv

exp_2_nbal_Radjenovic:
	@echo "Running eg__nbAL on Radjenovic sk + bl / minimal + preprocessed"
	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv
	$(PYTHON_VER) -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv

	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv


nbskal:
	$(PYTHON_VER) -m pip install scikit-learn pandas numpy
	@echo "Running nb_sk_al.py on all preprocessed and minimally processed datasets with n_pos=8,16,32"
	mkdir -p results_al
	for n in 8 16 32; do \
	  $(PYTHON_VER) -B nb_sk_al.py --input $(parent_dir)/test_data/preprocessed_Hall_num.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_Hall_preprocessed.csv; \
	  $(PYTHON_VER) -B nb_sk_al.py --input $(parent_dir)/test_data/preprocessed_Kitchenham_num.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_Kitchenham_preprocessed.csv; \
	  $(PYTHON_VER) -B nb_sk_al.py --input $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_Radjenovic_preprocessed.csv; \
	  $(PYTHON_VER) -B nb_sk_al.py --input $(parent_dir)/test_data/preprocessed_Wahono_num.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_Wahono_preprocessed.csv; \
	  $(PYTHON_VER) -B nb_sk_al.py --input $(parent_dir)/test_data/minimally_processed_Hall_num.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_Hall_minimally_processed.csv; \
	  $(PYTHON_VER) -B nb_sk_al.py --input $(parent_dir)/test_data/minimally_processed_Kitchenham_num.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_Kitchenham_minimally_processed.csv; \
	  $(PYTHON_VER) -B nb_sk_al.py --input $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_Radjenovic_minimally_processed.csv; \
	done

exp_diabetes_al:
	@echo "Running eg__nbAL on diabetes"
	python3 -B bl-added.py -M sklearn --nbAL $(parent_dir)/../moot/classify/diabetes.csv
	python3 -B bl-added.py -M bl --nbAL $(parent_dir)/../moot/classify/diabetes.csv

exp_diabetes:
	@echo "training and testing on all samples"
	@echo "bl_new"
	python3 -B bl-added.py -M sklearn --naive $(parent_dir)/../moot/classify/diabetes.csv
	@echo "bl_old"
	python3 -B bl-added.py -M bl --naive $(parent_dir)/../moot/classify/diabetes.csv

exp_diabetes_sk_al:
	for n in 8 16 32; do \
	  python3 -B nb_sk_al.py --input $(parent_dir)/../moot/classify/diabetes.csv --n_pos $${n} --repeats 20 --batch_size 1 --output results_al/results_uncertainty_gaussian_nb_$${n}_diabetes.csv; \
	done

exp_diabetes_sk_naive:
	@echo "Using sklearn GaussianNB"
	python3 -B nb_sk_naive.py --input $(parent_dir)/../moot/classify/diabetes.csv

test_full:
	@echo "Running eg__nbAL on Hall sk + bl / minimal + preprocessed"

	python3 -B bl-added.py -M bl --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num_copy.csv
	python3 -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num_copy.csv

test_parallelized:
	$(PYTHON_VER) -B bl_p.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num_copy.csv

test_baseline_p:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num_copy.csv

hall_sk_pre:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Hall_num.csv

hall_sk_min:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num.csv

wah_sk_pre:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Wahono_num.csv

wah_sk_min:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Wahono_num.csv

rad_sk_pre:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv

rad_sk_min:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv


# ADDING NEW SETS WITH INDIVIDUAL N_POS VALS

rad_sk_pre_16:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv 16

rad_sk_pre_32:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Radjenovic_num.csv 32

rad_sk_min_16:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv 16

rad_sk_min_32:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Radjenovic_num.csv 32

wah_sk_pre_16:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Wahono_num.csv 16

wah_sk_pre_32:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Wahono_num.csv 32

wah_sk_min_16:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Wahono_num.csv 16

wah_sk_min_32:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Wahono_num.csv 32

hall_sk_pre_16:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Hall_num.csv 16

hall_sk_pre_32:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/preprocessed_Hall_num.csv 32

hall_sk_min_16:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num.csv 16

hall_sk_min_32:
	$(PYTHON_VER) -B bl-added.py -M sklearn --nbAL $(parent_dir)/test_data/minimally_processed_Hall_num.csv 32