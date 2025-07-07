SCRIPTS = *.py

all:
	mkdir -p figures; for script in $(SCRIPTS); do echo "Running" $$script; python $$script; done

clean:
	rm figures/*.pdf
