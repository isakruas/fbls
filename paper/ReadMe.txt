FBLS: Preliminary Catalog of Exoplanet Transit Candidates Identified in TESS Photometry
Isak Paulo de Andrade Ruas, NROLABS
Version: <date>

Candidates from a blind transit search (FBLS, https://github.com/isakruas/fbls) of
TESS 2-minute SPOC light curves. Every signal went through automated vetting, a
difference-image pixel test and TRICERATOPS false positive probabilities; CTOI stars
were re-searched on every sector available. Preliminary: dispositions may change in
later versions.

Files
  ctoi.csv         One row per candidate submitted (or to be submitted) to ExoFOP as a
                   community TESS Object of Interest, in the ExoFOP CTOI upload columns.
  candidates.csv   Every vetted signal of every searched star, with its tier.
  figures.zip      One folder per CTOI signal, named <TIC>_<signal>, one PNG per panel:
                   light_curve, folded, transits, odd_even, period_maxima, orbit, pixels.

ctoi.csv columns (ExoFOP CTOI template; errors in the column after each value)
  TIC ID; Transit Epoch (BJD); Period (days); Depth ppm; Duration (hrs); Impact Param;
  Rad_p/Rad_s; a/Rad_s; Inclination (deg); Planet Radius (R_Earth); Insolation (Earth Flux);
  Equilibrium Temp (K); Fitted Stellar Density (g/cm^3); User Disposition; Notes.
  Geometry columns are empty when the trapezoid fit was not reliable.

candidates.csv columns
  tic                 TESS Input Catalog number
  signal              signal index within the star
  search              periodic (folded search) or single (unfolded single-event scan)
  tier                strong, candidate, inconclusive, rejected or duplicate, with the reason
  known, known_source, known_disposition
                      match to TOI, CTOI, confirmed planets, eclipsing binaries or TCEs (empty: new)
  period_days, epoch_bjd, duration_hours, depth_ppm, radius_earth
                      vetting values (epoch BJD_TDB; radius from the TIC stellar radius)
  detector_snr        signal-to-noise of the search statistic
  vetting_snr         depth over its error from the folded light curve
  transits            transits covered by the searched sectors
  planet_log_odds, model_preferred
                      planet-versus-binary evidence of the detector and whether planet won
  fpp, nfpp           TRICERATOPS false positive and nearby false positive probabilities
  pixel_offset_arcsec difference-image source offset from the target
  flags               vetting flags as name=fail|warn
