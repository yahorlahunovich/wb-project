# Report

## Features
Here we present used features:

**log distance** - distance for each intrachromosomal contact calculated with formula:
$$\frac{\log_2(d+1)+0.1}{0.1}$$
For each cell we computed probability distribution of such distances binned to intervals of lengths 0-50, 50-100, 100-150, 150-200, 200-250, 250-300.

**bins contacts fraction** - for chromosome splited to parts (bins) of length 4Mb we computed fraction of intrachromosomal contacts for each pair of bins.

**% near** - % of intrachromosomal contacts with log-dist from range 38-89.

**% mitotic** - % of intrachromosomal contacts with log-dist from range 90-109.

**average far contacts distance** - average of log-dists for contacts with log-dist greater than 98.

**transchromosomal contacts** - % of contacts between different chromosomes in a cell.

**raw replicate score** - average number of intrachromosomal contacts which starts before 50Mb.

**loop enrichment** - for each cell we computed number of loops in it (TODO ???)

**compartments** - TODO number for intrachromosomal contact between A and B (and for transchromosomal) - WHAT DOES THIS NUMBER MEAN?

**insulation score** - for chromosome binned to bins of length 50kb we calculated insulation score with window size 5. We averaged those values in each cell.

**average distance** - average contact length for each cell.

**distance standard deviation** - standard deviation of contact lengths for each cell.

**distance skewness** - skewness of contact lengths, for each cell.

**distance kurtosis** - kurtosis of contact lengths, for each cell.

**percentile distances** - percentiles 25, 75 and 95 of contacts lengths, for each cell.

**distance entropy** - entropy of log distance bins (0-50, 50-100, ..., 250-300), for each cell.