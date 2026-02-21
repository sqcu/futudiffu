
### Run the tests

When this environment has a GPU and a Python interpreter -- which it does --
do not propose "you could run this test to validate" as a future action item.
Run the test. Right now. A test that has been written but never executed is
not a test; it is a hypothesis about what a test might do. "Write test
effecting constraint X" means write it AND run it AND observe whether it
actually effects constraint X. End-to-end tests with real data are tech debt
until they have been run and had their own bugs shaken out.

If a test fails: that is useful information. Fix it or report it. Do not
leave it as a TODO.

### Persist test outputs (append-only)

Test outputs (rollout latents, metrics JSONL, rendered PNGs, timing data)
must be saved to disk, not printed-and-discarded. These artifacts are the
evidence base for cross-comparison across scripts, environments, and time.

Do not reflexively delete old test output directories because "tidying up"
feels productive. Test output is effectively append-only. The ability to
compare today's run against last week's run is worth more than a clean
working tree. Stale outputs age out naturally; prematurely deleted outputs
cannot be reconstructed if the environment that produced them no longer
exists (spot instances, hardware swaps, kernel upgrades).

Structure: `<test_name>_output/` directories at repo root or under a
configurable `--output-dir`. Filenames include enough context to be
self-describing (iteration count, timestamp, config hash, etc.).

### Defensive coding, bloatfiles, and statistically correct technical debt:

Tests exist in one place and in one form. A scripts dir, where a script runs an end to end test which uses a module to map input data to output data. Tests which cannot be confirmed or disconfirmed through functional mappings of data are a 'jelqing ring', 'gooning sesh', or 'bikeshedding affair', and prohibited by our style guide.

Unit tests are categorically forbidden. Unit tests are not verifiable and do not contribute to functional (end to end data mapping) verification of a subdomain or the grand domain of our deployed programs. There is no legitimate cause for a unit test, and all unit tests discovered in the repository are to be removed without explanation.

If you don't know what constitutes a functional mapping or end to end integration test for a specific task which seems to need tests, it is either 1: a fake problem, no tests are needed, write the real code and run it, 2: a fake abstraction or fake module or fake function which should not exist if it cannot be connected to any input, any output, or any application, or 3: a bad case of mesa-mis-management. someone should have used the 'ask user question tool' ten thousand tokens ago, probably, and the codebase is scatterd with inductive evidence *confirming* the user repeatedly and tangibly intervenes to make verification of tricky modules and behaviors tractable. (see i2i_off_policies\PINKIFY_cases; src_ii\reward_functions.py PINKIFY defs via a subagent if tangible examples of mapping-based-specification are needed.)