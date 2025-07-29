SCRIPTS = *.py

all: tikz/schematic.pdf
	mkdir -p figures; for script in $(SCRIPTS); do echo "Running" $$script; python $$script; done

tikz/schematic.pdf: tikz/schematic.tex
	cd tikz; pdflatex schematic.tex

figures/fig01.pdf: tikz/schematic.pdf
	cp tikz/schematic.pdf figures/fig01.pdf

figures/fig02.pdf: figures/verification_of_approximation.pdf
	cp figures/verification_of_approximation.pdf figures/fig02.pdf

figures/fig03.pdf: figures/timeseries_and_phase_plane.pdf
	cp figures/timeseries_and_phase_plane.pdf figures/fig03.pdf

figures/fig04.pdf: figures/bifurcation_diagram.pdf
	cp figures/bifurcation_diagram.pdf figures/fig04.pdf

figures/fig05.pdf: figures/critical_ecs_as_func_of_q10_chalf_ca0.pdf
	cp figures/critical_ecs_as_func_of_q10_chalf_ca0.pdf figures/fig05.pdf

figures/fig06.pdf: figures/jules_log_dCa_fit_many.pdf
	cp figures/jules_log_dCa_fit_many.pdf figures/fig06.pdf

figures/fig07.pdf: figures/jules_growth_rate_fit_many.pdf
	cp figures/jules_growth_rate_fit_many.pdf figures/fig07.pdf

number_figures: figures/fig01.pdf figures/fig02.pdf figures/fig03.pdf figures/fig04.pdf figures/fig05.pdf figures/fig06.pdf figures/fig07.pdf

clean:
	rm figures/*.pdf
