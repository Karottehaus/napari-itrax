# napari-itrax

A [napari](https://napari.org/) plugin for processing ITRAX X-ray fluorescence (XRF) scans and calibrating sediment core photographs.

The plugin provides two interfaces:

- **ITRAX Processing**: combine scan results with true-depth information, review element quality, and export processed tables and depth profiles.
- **Photo Ruler**: calibrate photographs and add photo-depth or core-depth rulers, individually or in batches.

## Installation

Requires **Python 3.11 or later**, **napari 0.6 or later**, and a Qt backend.

Create and activate an environment, then install the plugin from this repository:

```bash
conda create -n itrax python=3.12 pip
conda activate itrax
python -m pip install "napari[pyqt6]" "git+https://github.com/Karottehaus/napari-itrax.git"
```

The GitHub installation command requires Git. If you have already downloaded or cloned the repository, run the following from its root directory instead:

```bash
python -m pip install "napari[pyqt6]" .
```

Start napari:

```bash
napari
```

Open **ITRAX Processing** or **Photo Ruler** from napari's **Plugins** menu under **napari ITRAX**.

## ITRAX Processing

### Prepare the input

Select a folder containing your ITRAX scans. The plugin searches its subfolders for result files. It prefers re-evaluated `Results.txt` over instrument `result.txt` files and also accepts names such as `result_<name>.txt`.

Keep each scan's `document.txt` and associated images alongside the scan data when available. Instrument settings are read from `document.txt`; values entered in the interface supply missing settings. Automatic photo rulers require optical start and end coordinates.

A **True Depth** Excel workbook (`.xlsx` or `.xlsm`) maps scans to cores, sections, and depth coordinates. Use **Create True Depth template…** to generate a workbook from the scan folder, then complete and check it before processing.

| Column | Description |
| --- | --- |
| `Core` | Name used to identify the scan or section. |
| `CoreID` | Identifier grouping sections belonging to the same core. |
| `SectionID` | Section identifier within the core. |
| `StartID` | True depth, in millimetres, corresponding to `StartPosition`. |
| `StartPosition` | Scanner reference position in millimetres; if blank, the start coordinate from `document.txt` is used. |
| `Folder` | Optional folder mapping, populated by the template to associate rows with scan folders. |

Depth is calculated as:

```text
DepthID = position - StartPosition + StartID
```

The generated template leaves `StartID` blank for you to fill in. Review the generated core and section identifiers as well.

### Process and select elements

1. Choose the **Scan folder**, completed **True Depth** workbook, and **Output folder**.
2. Review instrument settings and choose whether to include XRF replicates and export photos with depth rulers or Ar-count overlays.
3. Choose Excel or CSV output and an invalid-point policy: `empty`, `keep`, or `leave out`. The suspect-point setting defaults to an Ar threshold factor of `1.5`; leave it empty to disable suspect flagging.
4. Click **Process scans**.
5. Review the element table, which reports zero counts, coefficient of variation, estimated noise, repeatability, and potential peak overlaps. Select the elements to include in normalization and centered log-ratio (CLR) calculations. Element selections can be saved and loaded.
6. In **Export**, configure element ratios and log ratios, optional incoherent/coherent scattering output, and detailed plot pages. Click **Write element files and plots**.

Quality flags support element selection; review them in the context of your measurements. Requested ratios must use elements present in the run.

### Outputs

Results are written to `<scan-folder-name>_Results` inside the selected output folder. Depending on your choices, they include:

- Per-scan XRF tables and replicate results.
- Selected element counts (`slct`), normalized percentages (`norm`), and CLR-transformed values.
- Element ratios, log ratios, and depth-profile plots.
- Core photographs, depth-ruler images, and Ar-count overlays.
- A PDF results guide and `run_info.txt` recording settings and processing notes.

Excel exports can also organize each scan's treatments into separate tabs in one workbook.

Normalization expresses each selected element as a percentage of the sum of selected elements. Before normalization and CLR calculation, zeros are replaced with half the smallest positive value for that element within the core, falling back to the run-wide value when necessary. The chosen element set therefore affects the transformed results.

## Photo Ruler

Use **Photo Ruler** to add calibrated rulers to photographs independently of XRF processing.

1. Click **Choose photos…** and select the images to process.
2. Set the location of the core top in the original photograph: left, right, top, or bottom.
3. Choose a scale source: DPI, millimetres per pixel, full photo length, an ITRAX document, or an interactive measurement.
4. For interactive calibration, enter the known distance in millimetres, click **Show selected photo and markers**, and position the two markers along the core axis. Apply the measurement to the selected photo or to all photos sharing the same scale.
5. Enable the photo-depth ruler, core-depth ruler, or both, and supply core-depth information where required. Choose vertical output if desired.
6. Select an output folder and click **Export ruler photos**.

ITRAX documents can be assigned per photograph, shared across the batch, or located automatically. Interactive calibration uses the distance along the core axis rather than the diagonal distance between markers.

## Contributing

Bug reports and contributions are welcome. To set up a development installation from a local checkout:

```bash
python -m pip install -e ".[testing]"
```

Run the tests and code checks from the repository root:

```bash
pytest
ruff check src tests
mypy src/napari_itrax
```

For tests on a machine without a display, set `QT_QPA_PLATFORM=offscreen`.

## Issues

Please [open an issue](https://github.com/Karottehaus/napari-itrax/issues) with a description of the problem, steps to reproduce it, your operating system, Python and napari versions, and any error message. For data-related problems, include a small example that can be shared publicly.

## Author

Developed by Hanyu Xu, University of Bern.

## License

Distributed under the [MIT License](LICENSE).
