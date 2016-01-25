$ yatsm pixel --help
Usage: yatsm pixel [OPTIONS] <config> <px> <py>

Options:
  --band <n>           Band to plot  [default: 1]
  --plot [TS|DOY|VAL]  Plot type  [default: TS]
  --ylim <min> <max>   Y-axis limits
  --style <style>      Plot style  [default: ggplot]
  --cmap <cmap>        DOY/VAL plot colormap  [default: viridis]
  --embed              Drop to (I)Python interpreter at various points
  --seed TEXT          Set NumPy RNG seed value
  --algo_kw TEXT       Algorithm parameter overrides
  --help               Show this message and exit.
