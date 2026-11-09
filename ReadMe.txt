FBLS: Catalog of Exoplanet Transit Candidates Identified in TESS Photometry
Isak Paulo de Andrade Ruas, NROLABS
Version: 2026-11-08
DOI (all versions): 10.5281/zenodo.24026413

Candidates from a blind transit search of TESS 2-minute SPOC light curves with FBLS
(Family Box Least Squares, https://github.com/isakruas/fbls), a GPU detector:
per-duration detrending families, correlated-noise scaling of the amplitude
statistic, trapezoidal shapes, a multi-pass search with masking of accepted signals, a
single-event scan and a joint planet/binary/noise evidence at every peak. Every signal went
through automated vetting (depth and SNR, odd/even, secondary eclipse, shape, duration
against stellar density, centroid, crowding, and the Kepler Robovetter / LEO vetter flux
tests: ModShift uniqueness, SWEET sinusoidal variability, individual-event consistency,
gaps, model comparison) and a difference-image pixel test; CTOI stars were re-searched on
every sector available. Dispositions may change in later versions.

Files
  ctoi.csv         One row per candidate submitted (or to be submitted) to ExoFOP as a
                   community TESS Object of Interest, in the ExoFOP CTOI upload columns.
  candidates.csv   Every vetted signal of every searched star, with its tier.
  figures.zip      One folder per CTOI signal, named <TIC>_<signal>, with the data of each
                   panel as CSV (light_curve, folded with the trapezoid model and parity,
                   transits, event.json; pixel difference, error and out-of-transit grids).

ctoi.csv columns (ExoFOP CTOI template; errors in the column after each value)
  TIC ID; Transit Epoch (BJD); Period (days); Depth ppm; Duration (hrs); Impact Param;
  Rad_p/Rad_s; a/Rad_s; Inclination (deg); Planet Radius (R_Earth); Insolation (Earth Flux);
  Equilibrium Temp (K); Fitted Stellar Density (g/cm^3); User Disposition; Notes.
  Geometry columns are empty when the trapezoid fit was not reliable.

candidates.csv columns
  tic                 TESS Input Catalog number
  signal              signal index within the star
  search              periodic (folded search) or single (unfolded single-event scan)
  tier                candidate, unconfirmed (fewer than 3 transits), inconclusive, rejected or
                      duplicate, with the reason
  known, known_source, known_disposition
                      match to TOI, CTOI, confirmed planets, eclipsing binaries or TCEs (empty: new)
  period_days, epoch_bjd, duration_hours, depth_ppm, radius_earth
                      vetting values (epoch BJD_TDB; radius from the TIC stellar radius)
  detector_snr        signal-to-noise of the search statistic
  vetting_snr         depth over its error from the folded light curve
  transits            transits covered by the searched sectors
  planet_log_odds, model_preferred
                      planet-versus-binary evidence of the detector and whether planet won
  pixel_offset_arcsec difference-image source offset from the target
  flags               vetting flags as name=fail|warn
