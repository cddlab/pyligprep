# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

リガンドPDB → 水素付加 → PySCF/GPU4PySCFで構造最適化 → HF/6-31G* ESP → 内蔵の二段階RESP →
AmberTools 26（antechamber / parmchk2 / tleap）でGAFF2パラメータ（`.mol2` / `.prep` / `.frcmod`）を出力するCLIパッケージ。
対象は単一分子・単一残基・単一配座の H/C/N/O/F/P/S/Cl のみ。ユーザー向けドキュメントは README.md（英語、`pyproject.toml` の `readme`）と README_ja.md（日本語）の2本立てで、
片方を変えたらもう片方も同じ内容に更新する。docs/validation.md は日本語のみ。
ライセンスは GPL-3.0-only（`resp.py` が GPL-3.0-only の AmberTools PyRESP からの移植のため。「or later」にはできない）。

## コマンド

```bash
uv sync --extra cuda12        # CPUのみ（macOS含む）なら uv sync
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync ty check src tests
uv run --no-sync pytest -q
uv run --no-sync pytest tests/test_charge_rounding.py::<test_name> -q   # 単一テスト
uv build
```

- テストマーカー `gpu` / `amber` / `reference` は環境変数 `PYLIGPREP_TEST_GPU=1` / `PYLIGPREP_TEST_AMBER=1` /
  `PYLIGPREP_TEST_REFERENCE=1` がない限り `tests/conftest.py` で自動スキップされる。
  AmberToolsの場所は `PYLIGPREP_AMBER_BIN`、`AMBERHOME/bin`、PATH上の `antechamber` の順で解決。
- `reference` テストは AmberTools 同梱の元の PyRESP / `pyresp_gen.py` を比較用オラクルとしてsubprocessで呼ぶ（実行時依存ではない）。
- GPUテストはSlurm経由: `sbatch scripts/test_gpu_slurm.sh`（partition `q4`、`--cov` を使うため `pytest-cov` が別途必要）。
  `scripts/run_sin_slurm.sh` はSIN専用の `run` 例（作業ディレクトリが入力ディレクトリ、GPU 1基、
  1タスク×8 CPUを要求し、`sbatch` で投入したスクリプトから `pyligprep` を直接起動）。module名・パスはこの計算機固有。
- AmberToolsの探索: CLIは `--amber-bin` かPATHのみ。`PYLIGPREP_AMBER_BIN` / `AMBERHOME` を見るのはテストだけ。
- AmberToolsモジュールをロードした環境では `unset PYTHONPATH` してからvenvを使う（AmberTools側のPythonが混入するため）。
- 通常テストはCPUでHF/ESPまで実行するが、CPUでの構造最適化は実走しない。
- macOS（Apple Silicon、CPUのみ）は副次的なサポート対象: `uv sync`、AmberToolsはHomebrew
  `brewsci/bio` の `ambertools`（`AMBERHOME=/opt/homebrew/opt/ambertools`）。PyPIのmacOS用PySCF wheelは
  OpenMPなしのため、`pyproject.toml` の `[tool.uv.sources]`（darwinのみGitHubのv2.15.0アーカイブ。PyPI sdistは
  `libxcfun.patch` 欠落でビルド不可）と `[tool.uv.extra-build-variables]`（Homebrew LLVMのclang、要 `brew install llvm`）で
  OpenMP付きにソースビルドする。Linuxはwheelのまま。PySCFを更新する際はこのURLも書き換える。E2E確認は `examples/sin_h.pdb`（charge −2、SMILES `O=C([O-])CCC(=O)[O-]`）で `run` する。

## アーキテクチャ

### ステージ／キャッシュ機構（`artifacts.py` の `Stage`）
`opt` / `resp` / `export` / `run` はすべて `Stage` コンテキストマネージャ上で動く。これがこのコードベースの中核:

- 出力構成: `-o D` の直下には成果物と `versions.json` だけを置き、それ以外はすべて `D/tmp/` に置く（`publish=True`）。
  `run` の子ステージは `publish=False` で、`D/tmp/{opt,resp,export}` 自体が作業ディレクトリ（その下に `tmp/` は作らない）。
- 作業ディレクトリは `state.json`（schema=2: stage、status、molecule、settings、fingerprint、versions、steps、
  published、result）と `.lock`（`fcntl.flock` で排他）を持つ。試行履歴（attempts）はなく、再開は同じ場所で続きから行う。
  旧形式（`manifest.json` / `attempts/`）は `check_resumable()` が検出し、新しい出力先での実行を案内するエラーにする。
- `fingerprint = digest(molecule + settings + 依存パッケージのversions())`。完了済みかつfingerprint一致なら
  `stage.cached` に検証済み結果が入り、計算をスキップする（成果物と `versions.json` は再公開）。依存バージョンが変わるとキャッシュは無効。
- `complete(result, deliverables=, discard=, binaries=)` は `discard`（`*.chk`）を削除してから成果物を直下へ公開し、
  `versions.json` を書く。`completed_result()` は `result["files"]` の全ハッシュを検証する（手編集は「改変」として拒否）。
- `Stage.publish()` は直下への公開で、`published` に記録済みのハッシュ以外の既存ファイル（手編集）とsymlinkは上書きしない。
- 同じ分子での再開は `stage.resumed=True` と `stage.prior_settings`（前回の設定）で判定し、作業ディレクトリ内の
  `last.json` / `*.chk` を再開に使う。`opt` / `run` では molecule が変わった再開はエラー。`resp` / `export` は
  molecule 変更時に `.lock` 以外の作業ディレクトリを消去して新規に計算する（別分子の `hf.chk` を初期値にしないため）。
- `run` は成功後に `D/tmp/` を丸ごと削除する（`--keep-tmp` で保持、失敗時は常に保持）。削除済みの完了ディレクトリへの
  `--resume` は `check_resumable()` が案内つきで拒否する。

### RESPのサブステップキャッシュ（`charges.py`）
`resp` ステージ内で HF → ESP → stage1 → stage2 を `state.json` の `steps`（fingerprint＋ファイルハッシュ）に個別に記録し、
`stage.reusable()` が一致を確認して同じ作業ディレクトリのファイルを再利用する（再計算前に `stage.forget()`）。
フィット設定だけの変更ではESPを再利用。`hf.chk` は完了時に削除されるため、HFの再利用は未完了の工程を再開する場合だけ。
ESP座標はBohr／原子単位、`Molecule` の座標はÅ（`esp.BOHR` で変換）。

### データモデルと工程間の受け渡し
- `model.Molecule` / `Atom` はfrozen dataclass。原子名・順番・残基情報・形式電荷・全桁座標を保持し、`to_dict()` / `from_dict()`
  で `state.json` に保存される。工程間はPDB（丸め座標）ではなくこのdictで渡す。`from_dict` は常に `validate()` する。
- `chemistry.read_pdb` は先頭が数字の原子名（`1H1` など）を拒否する（antechamberがMOL2作成時に元素を推定できないため）。
- PDB入力時の原子名・順番は最終MOL2/PREPまで保持され、`amber.py` が出力を読み戻して名前・順番・電荷・結合を検証する。
- `hadd`（`chemistry.add_hydrogens`）は `<name>.pyligprep.json` サイドカーに結合次数・形式電荷を保存。PDBのハッシュが
  変わるとサイドカーは無効。`run` は水素を付加しない。
- テンプレートなしの `hadd` は Open Babel（必須依存の `openbabel-wheel`）で結合次数を推定する。Open Babel は
  GPL-2.0-only で GPL-3.0 と非互換のため、`openbabel` は**決して import しない**。`chemistry.obabel_command()` が
  wheel 同梱の `obabel` バイナリを import なしで探し、`BABEL_LIBDIR` / `BABEL_DATADIR` を付けて別プロセスで実行する
  （`test_openbabel_hadd_without_template` が `sys.modules` に `openbabel` がないことを検査）。

### CLIとパイプライン
- `cli.py` は量子化学系モジュールを関数内で遅延import（`doctor` / `check` / `export` でPySCF/CuPyを読み込まないため）。
  この方針を崩さない。
- バージョンは `pyproject.toml` の `project.version` が唯一の定義元。`pyligprep.__version__` はインストール済みメタデータ
  （未インストールなら隣の `pyproject.toml`）から読むので、番号を直書きしない。変更後は `uv sync` で再インストールする
  （`artifacts.versions()` もメタデータを読み、fingerprintに入る）。`cli._Parser` が全パーサに `-V/--version` と
  `--help` 先頭のバージョン行を付けるため、サブコマンドは `argparse.ArgumentParser` ではなくこのクラスで作る
  （`add_parser` は親のクラスを引き継ぐ）。
- `--resume` は `tmp/state.json` の settings を読み込み（`artifacts.load_state()`）、CLIで明示した値だけ上書き（`_options()`）。charge/multiplicity の変更は拒否。
- Slurmジョブ内の `--threads` 省略時だけは、再開時も現在の割り当てから決定し直す（`cli._quantum_options()`）。
  `resources.py` が `SLURM_CPUS_PER_TASK` とCPU affinityを参照し、複数タスクと明示値の超過を拒否する。
  CPU affinity自体は変更しない。Python APIの `QuantumOptions.threads` は引き続き明示整数（既定8）。
- `pipeline.run_pipeline` は `run` ステージを `reuse_complete=False` で開き、子ステージ（`opt`/`resp`/`export`）を毎回
  再訪する。各子ステージが自身のキャッシュを判定する。CLIと同じ関数（`optimize` / `calculate_charges` / `export_parameters`）を使う。
- 想定内の失敗は `errors.py` の `PreparationError` 派生（`InputError` / `CalculationError` / `DependencyError`）で送出し、
  `main()` が終了コード2で表示する。

### 出力電荷の丸め（`charge_rounding.py`）
電荷は小数5桁（ROUND_HALF_UP）にし、1e-5 e単位の整数演算で総和を指定電荷に厳密一致させる。RESP第2段階の等価拘束と、
第2段階で固定された原子間の第1段階の等価拘束を維持する。両立不能な場合は明示的にエラー。
ポリシー名 `POLICY` はexport settingsに入るため、挙動を変えたらポリシー名も更新する（既存exportのキャッシュ無効化のため）。

### D3(BJ)分散補正のバックエンド（`dispersion.py`）
既定の `B3LYP-D3BJ` はPySCFが `pyscf.dispersion`（pyscf-dispersion、Linux wheelのみ）を呼ぶ。macOSでは
`pyproject.toml` のマーカーで代わりに `dftd3`（simple-dftd3）が入り、`quantum.make_method()` が呼ぶ
`ensure_dispersion()` が `sys.modules["pyscf.dispersion"]` に互換シムを登録する（同じs-dftd3パラメータ）。
D4はmacOSでは `DependencyError`。`doctor` の `dispersion.d3_backend` で使われるバックエンドを確認できる。

### その他の不変条件
- `--device auto` はGPU初期化失敗時のみCPUにフォールバック。GPU開始後のSCF未収束やOOMをCPU再実行で隠さない。
- 成果物はJSON / npz（`allow_pickle=False`）/ テキストのみ。pickleは使わない。JSONは `files.write_json` で原子的に書く。
- `--resname` は `[A-Za-z][A-Za-z0-9]{0,2}`、大小文字保持。出力prefixと残基名にのみ影響し、RESP結果・原子名は不変。

## コードスタイル

- 全Pythonファイルは `ruff format` 済み（2スペースインデント、`line-length = 120`）。`ruff check` と `ty check` は
  指摘ゼロを維持する。
- パッケージ内のimportは `from pyligprep.files import ...` のような絶対importに統一（相対importはRuffのTID252で禁止、
  `ban-relative-imports = "all"`）。
- `zip()` は長さが一致すべき組に `strict=True` を付ける（B905）。Options系のfrozen dataclassを引数の既定値に使うため、
  B008の例外として `[tool.ruff.lint.flake8-bugbear] extend-immutable-calls` に登録している。新しいOptionsを足したら追記する。
- 環境依存のimport（`cupy` / `gpu4pyscf` / `pyscf.dispersion`）は `[tool.ty.analysis] allowed-unresolved-imports` で許可。
  RDKitの `AllChem.EmbedMolecule` 等はC拡張の再エクスポートでtyが解決できないため、`rdDistGeom` など定義元モジュールを使う。
- 検証作業の生ログ・生成物は `validation/`（gitignore済み）に置く。検証結果の記録は `docs/validation.md` に追記する。
