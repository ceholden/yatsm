TODO
====

# Model Parameters
## The threshold on multitemporal noise screening

Should customize this to handle more or less noise in the timeseries as the amount screened can impact change detection

## The time to refit the model

`if abs(self.X[self.here, 1] - self.trained_date) > self.ndays:`

Change to

`if abs(self.X[self.here, 1] - self.trained_date) > self.retrain_time:` 

Should help:
    - Calculation time
    - Ability to find slow, gradual changes that might be missed if the expectation (e.g., the regression model) adapts too quickly

## Timeseries Drivers

Delete them! Recode into a `utils` for grabbing a single pixel
