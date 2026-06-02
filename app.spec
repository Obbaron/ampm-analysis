# app.spec — PyInstaller spec file for ampm-analysis GUI
# Usage: pyinstaller app.spec
# Output: dist/ampm-analysis/ (folder with exe)

import sys

block_cipher = None

# Platform-specific icon
if sys.platform == 'darwin':
    app_icon = 'assets/ampm.icns'
else:
    app_icon = 'assets/ampm.ico'

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('assets/ampm.ico', 'assets'),
        ('assets/ampm.icns', 'assets'),
    ],
    hiddenimports=[
        # ampm package — lazy imports inside worker thread
        'ampm',
        'ampm.config',
        'ampm.loading',
        'ampm.masking',
        'ampm.assignment',
        'ampm.clustering',
        'ampm.cov',
        'ampm.views',
        'ampm.actions',
        'ampm.setup_build',
        'ampm.cluster_cache',
        'ampm.mask_cache',
        'ampm.parts',
        'ampm.stats',
        # View modules — discovered dynamically, invisible to PyInstaller
        'ampm.views.bar',
        'ampm.views.contour',
        'ampm.views.cov_summary',
        'ampm.views.k_distance',
        'ampm.views.kde',
        'ampm.views.layer_viewer',
        'ampm.views.scatter_2d',
        'ampm.views.scatter_3d',
        'ampm.views.single_layer',
        # Core dependencies
        'polars',
        'pyarrow',
        'numpy',
        'scipy',
        'scipy.spatial',
        'scipy.sparse',
        'scikit-learn',
        'sklearn',
        'sklearn.cluster',
        'sklearn.neighbors',
        'sklearn.preprocessing',
        # Geometry
        'trimesh',
        'shapely',
        'shapely.geometry',
        'rtree',
        'networkx',
        # Trimesh backends — loaded at runtime
        'embreex',
        'lxml',
        'mapbox_earcut',
        'manifold3d',
        'svg.path',
        'pycollada',
        # Polars backend
        'fastexcel',
        # Plotting
        'plotly',
        'plotly.express',
        'plotly.graph_objects',
        # Qt
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ampm-analysis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=app_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ampm-analysis',
)
