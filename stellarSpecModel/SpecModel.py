import os
import json
import hashlib
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from tqdm.auto import tqdm
from spectool import pyrebin
from .SpecGrid import SpecGrid
from .excepts import AliasAlreadyExistsError
from . import config
from . import registry
import logging
logger = logging.getLogger(__name__)


modelRegistry = registry.get_registry()


class SpecModel:
    def __init__(self, grid: SpecGrid):
        """
        :param grid: SpecGrid, including data and meta info
        """
        self.grid = grid

    @classmethod
    def list_models(cls, kind: str = None) -> list:
        """
        列出注册表中的模型。
        
        :param kind: 筛选特定类型的模型，支持 'official', 'local', 'derived', 'alias'。
                     如果不传，则列出全部模型。
        :return: 包含模型信息的字典列表。
        """
        # 每次列出前，刷新一下底层物理文件的真实存在状态
        modelRegistry.refresh_status()
        
        if kind:
            valid_kinds = ["official", "local", "derived", "alias"]
            if kind not in valid_kinds:
                raise ValueError(f"Invalid kind '{kind}'. Must be one of {valid_kinds}")
            models = [m for m in modelRegistry.list_all_models() if m.get("kind") == kind]
        else:
            models = modelRegistry.list_all_models()

        # 为了方便用户在终端/Notebook查看，直接打印一张漂亮的简表
        if not models:
            print(f"No models found in registry for kind: {kind or 'all'}")
            return models

        print(f"{'Model ID':<30} | {'Kind':<10} | {'Status':<10} | {'Target/Parent':<20}")
        print("-" * 78)
        for m in models:
            m_id = m.get("id", "")
            m_kind = m.get("kind", "")
            m_status = m.get("status", "")
            # 对于 alias 展示 target，对于 derived 展示 parent
            m_extra = m.get("target", m.get("parent", "")) 
            print(f"{m_id:<30} | {m_kind:<10} | {m_status:<10} | {m_extra:<20}")
            
        return models

    @classmethod
    def get(cls, model_id: str, lazy: bool = True):
        """
        通过注册表中的 Model ID 加载模型。
        会自动穿透 alias 别名，寻找真实的物理文件。
        """
        # 1. 检查注册表中是否存在该 ID
        if not modelRegistry.exists(model_id):
            raise KeyError(
                f"Model '{model_id}' not found in registry. "
                "Use SpecModel.list_models() to see available models."
            )
            
        # 2. 检查状态
        record = modelRegistry.get(model_id)
        if record.get("status") == "remote":
            raise RuntimeError(
                f"Model '{model_id}' is marked as 'remote'. "
                "You need to download it first before loading."
            )
        elif record.get("status") == "missing":
            raise FileNotFoundError(
                f"The physical file for model '{model_id}' is missing from the disk."
            )
            
        # 3. 通过 Registry 获取绝对物理路径 (底层会自动处理 alias 的递归透传)
        absolute_path = modelRegistry.get_absolute_path(model_id)
        
        if absolute_path is None or not absolute_path.exists():
            raise FileNotFoundError(f"Resolved path for model '{model_id}' does not exist: {absolute_path}")
            
        # 4. 调用已有的 load 方法加载物理文件
        logger.info(f"Loading model '{model_id}' from {absolute_path}")
        return cls.load(absolute_path, lazy=lazy)

    @classmethod
    def load(cls, filepath, lazy=True):
        """
        class method: direct loading from HDF5 file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Derived grid file not found at {filepath}")
        
        # 依赖 SpecGrid 自身的 HDF5 读取能力
        loaded_grid = SpecGrid.from_hdf5(filepath, lazy=lazy)
        return cls(loaded_grid)

    def _generate_cache_key(self, select: dict, wavelength: dict) -> str:
        """
        Generate a unique hash fingerprint based on the normalized select and wavelength.
        """
        # Normalize select
        norm_select = {}
        for k in sorted(select.keys()):
            val = select[k]
            # Convert to list to ensure deterministic JSON serialization
            if isinstance(val, (tuple, list, np.ndarray)):
                norm_select[k] = np.asarray(val).tolist()
            else:
                norm_select[k] = val

        norm_wave = {}
        for parm in wavelength:
            value = wavelength[parm]
            if parm == 'range':
                norm_wave[parm] = list(value)
            elif isinstance(value, np.ndarray):
                norm_wave[parm] = np.asarray(value).tobytes()
            else:
                norm_wave[parm] = value

        fingerprint = {
            "selection": norm_select,
            "wavelength": norm_wave
        }
        fingerprint_str = json.dumps(fingerprint, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode('utf-8')).hexdigest()

    def _valid_wavelength(self, wavelength: dict | None = None):
        if wavelength is None:
            wave_range = (np.min(self.grid.wave), np.max(self.grid.wave))
            flag_resample = False
            method = 'custom'
            new_wave = None
            return wave_range, flag_resample, method, new_wave

        wave_range = wavelength.get("range")
        method = wavelength.get("method")
        step = wavelength.get("step")
        grid = wavelength.get("grid")

        # grid cannot coexist with other wavelength-generation options
        if grid is not None:
            if wave_range is not None:
                raise ValueError("'grid' cannot be used together with 'range'.")
            if method is not None or step is not None:
                raise ValueError("'grid' cannot be used together with 'method' or 'step'.")

            wave_range = (np.min(grid), np.max(grid))
            flag_resample = True
            return wave_range, flag_resample, method, grid

        # method and step must appear together
        if (method is None) != (step is None):
            raise ValueError("'method' and 'step' must be specified together.")

        if wave_range is None:
            wave_range = (np.min(self.grid.wave), np.max(self.grid.wave))
            
        # validate method
        if method is None:
            flag_resample = False
            method = 'custom'
            new_wave = None
            return wave_range, flag_resample, method, new_wave
        elif method == 'linear':
            min_w, max_w = wave_range
            new_wave = np.linspace(min_w, max_w, num=int((max_w - min_w) / step) + 1, endpoint=True)
            flag_resample = True
            return wave_range, flag_resample, method, new_wave
        elif method == 'log':
            min_w, max_w = wave_range
            log_min_w, log_max_w = np.log(min_w), np.log(max_w)
            log_new_wave = np.linspace(log_min_w, log_max_w, num=int((log_max_w - log_min_w) / step) + 1, endpoint=True)
            new_wave = np.exp(log_new_wave)
            flag_resample = True
            return wave_range, flag_resample, method, new_wave
        else:
            raise ValueError(f"Unsupported resampling method '{method}'.")

    def derive(
        self,
        select: dict | None = None,
        wavelength: dict | None = None,
        *,
        alias: str | None = None,
        cache_dir: str | Path | None = None,
        overwrite: bool = False,
        progress: bool = False,
    ) -> "SpecModel":
        """
        Create a derived spectral model from the current model.

        The ``derive`` method creates a new spectral grid by selecting a
        subset of the original parameter grid and/or modifying the wavelength
        sampling. It does not generate new physical models or interpolate
        along model parameter axes.

        Model parameters (e.g. ``teff``, ``logg``, ``feh``) can only be
        selected from existing grid points. The wavelength axis is treated
        differently and supports both wavelength selection and resampling.

        Parameters
        ----------
        select : dict, optional
            Selection rules for model parameters.

            The dictionary keys must correspond to existing parameter axes
            in the current model grid. Each parameter selection can be
            specified either as a range or as an explicit list of grid values.

            Examples
            --------
            Select a parameter range:

            >>> model.derive(
            ...     select={
            ...         "teff": (4500, 7000),
            ...         "logg": (2.0, 5.0)
            ...     }
            ... )

            Select specific existing grid points:

            >>> model.derive(
            ...     select={
            ...         "teff": numpy.ndarray([4500, 5000, 5500]),
            ...         "logg": (3.0, 4.0)
            ...     }
            ... )

            Notes
            -----
            - Parameters not included in ``select`` are preserved.
            - No interpolation is performed on model parameter axes.
            - Different spectral model families may contain different
            parameters. For example, a white dwarf model may only contain
            ``teff`` and ``logg``, while a stellar atmosphere model may
            additionally contain ``feh`` or ``alphaFe``.

        wavelength : dict, optional
            Operations applied to the wavelength axis.

            Supported operations include:

            ``range``
                Select a wavelength interval while preserving the original
                wavelength sampling.

                Example:

                >>> wavelength={
                ...     "range": (4000, 8000)
                ... }

            ``resample``
                Generate a regular wavelength grid using a predefined sampling
                scheme.

                Example (linear sampling):

                >>> wavelength={
                ...     "resample": {
                ...         "method": "linear",
                ...         "step": 1.0
                ...     }
                ... }

                Example (logarithmic sampling):

                >>> wavelength={
                ...     "resample": {
                ...         "method": "log",
                ...         "step": 1e-4
                ...     }
                ... }

                Example (custom sampling):

                >>> wavelength={
                        "resample": numpy.linspace(4000, 8000, 1000)
                    }

            Notes
            -----
            The wavelength operation order is:

            1. Crop the wavelength range (if specified).
            2. Generate the target wavelength grid.
            3. Interpolate spectra onto the new wavelength grid.

            Unlike model parameters, wavelength sampling can be changed.

        alias : str, optional
            User-defined name for the derived model.

            The alias is stored in the model metadata and can be used for
            identification in model registries or cache systems.

        cache_dir : str or pathlib.Path, optional
            Directory used to store the derived model.

            If provided, the derived model will be saved in this directory.
            Existing derived models may be reused if the same derivation
            configuration is found in the cache.

            If ``None``, the default package cache directory is used.

        overwrite : bool, default=False
            Whether to overwrite an existing cached derived model.

            If ``False``, an existing cached model with the same derivation
            hash may be reused.

            If ``True``, the derived model will be regenerated and the
            existing cached file will be replaced.

        Returns
        -------
        SpecModel
            A new derived spectral model.

            The returned model contains provenance information describing
            the parent model and all derivation operations.

        Raises
        ------
        ValueError
            If an invalid parameter name, parameter value, or wavelength
            operation is provided.

            Examples
            --------
            >>> model.derive(
            ...     select={"feh": (-1, 0)}
            ... )
            ValueError:
                Parameter 'feh' is not available.
                Available parameters: ['teff', 'logg']

        Notes
        -----
        The derived model keeps a complete record of its origin:

        Examples
        --------
        >>> derived.metadata["derived_from"]
        {
            "parent": "MARCS",
            "selection": {
                "teff": [4500, 7000],
                "logg": [2, 5]
            },
            "wavelength": {
                "range": [4000, 9000],
                "resample": {
                    "method": "log",
                    "step": 1e-4
                }
            }
        }

        This provenance information is used for:

        - reproducibility,
        - cache hashing,
        - model registry management,
        - future plugin extensions.
        """

        select = select or {}
        wavelength = wavelength or {}

        if cache_dir is None:
            active_cache_dir = Path(config.cache_PATH).expanduser()
        else:
            active_cache_dir = Path(cache_dir).expanduser()
        
        active_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_hash = self._generate_cache_key(select, wavelength)
        parent_id = self.grid.metadata.get('id')
        derived_model_id = f"{parent_id}_derived_{cache_hash}"
        cache_filename = f"{derived_model_id}.h5"
        cache_filepath = active_cache_dir / cache_filename

        if alias:
            if modelRegistry.exists(alias) and not overwrite:
                raise AliasAlreadyExistsError(
                    f"Alias '{alias}' already exists in registry. Pass overwrite=True to overwrite."
                )

        if cache_filepath.exists() and not overwrite:
            logger.info(f"Cache hit! Loading derived grid from {cache_filepath}")
            derived_model = self.__class__.load(cache_filepath)
            if not modelRegistry.exists(derived_model_id):
                modelRegistry.add_derived(
                    model_id=derived_model_id, 
                    parent_id=parent_id, 
                    filename=cache_filename,
                    # parameters={"select": select, "wavelength": wavelength}
                )
                modelRegistry.save()
            if alias:
                if modelRegistry.exists(alias):
                    modelRegistry.update(alias, {"target": derived_model_id})
                else:
                    modelRegistry.add_alias(alias, derived_model_id)
                modelRegistry.save()
            return derived_model

        # check the select par
        select_values = {}
        for param, val in select.items():
            if param not in self.grid.axis_names:
                raise ValueError(
                    f"Parameter '{param}' is not available. "
                    f"Available parameters: {list(self.grid.axis_names)}"
                )
            if isinstance(val, (list, np.ndarray)):
                valid_pts = set(self.grid.axes[param])
                invalid_pts = set(np.asarray(val)) - valid_pts
                if invalid_pts:
                    raise ValueError(f"Grid points {invalid_pts} for '{param}' do not exist.")
                select_values[param] = np.asarray(val)
            elif isinstance(val, tuple):
                if len(val) != 2:
                    raise ValueError(f"Grid point range for '{param}' must be a tuple of (min, max).")
                min_pt, max_pt = val
                valid_pts = set(self.grid.axes[param])
                arg = (self.grid.axes[param] >= min_pt) & (self.grid.axes[param] <= max_pt)
                select_value = self.grid.axes[param][arg]
                if len(select_value) == 0:
                    raise ValueError(f"Grid point range for {param} ({min_pt}, {max_pt}) does not match any existing grid points.")
                select_values[param] = select_value
            else:
                raise ValueError(f"Invalid selection value for '{param}': {val}. Must be a tuple, list, or ndarray.")

        wave_range, flag_resample, method, new_wave = self._valid_wavelength(wavelength)

        slice_indices = []
        new_axes = {}

        for param in self.grid.axis_names:
            axis_vals = self.grid.axes[param]
            if param in select_values:
                val = select_values[param]
                mask = np.any([np.isclose(axis_vals, v) for v in val], axis=0)
            else:
                mask = np.ones_like(axis_vals, dtype=bool)
            keep_idx = np.where(mask)[0]
            slice_indices.append(keep_idx)
            new_axes[param] = axis_vals[keep_idx]

        wave_vals = self.grid.wave
        wave_mask = (wave_vals >= wave_range[0]) & (wave_vals <= wave_range[1])
        wave_keep_idx = np.where(wave_mask)[0]

        # flux_slice_tuple = tuple(slice_indices + [wave_keep_idx])
        global_indices = slice_indices + [wave_keep_idx]
        mask_slice_tuple = tuple(slice_indices)

        bounding_box_slices = []
        local_indices = []

        for idx_array in global_indices:
            min_idx, max_idx = int(np.min(idx_array)), int(np.max(idx_array))
            bounding_box_slices.append(slice(min_idx, max_idx + 1))
            local_indices.append(idx_array - min_idx)
                
        bounding_box_slices = tuple(bounding_box_slices)

        sub_flux_tensor = self.grid.flux_tensor[bounding_box_slices]
        cropped_flux = sub_flux_tensor[np.ix_(*local_indices)]
        
        cropped_mask = self.grid.valid_mask[np.ix_(*mask_slice_tuple)]
        cropped_wave = wave_vals[wave_keep_idx]
        grid_pars = self.grid.grid_parameters
        cropped_grid_pars = {param: grid_pars[param][np.ix_(*mask_slice_tuple)] for param in grid_pars}

        if new_wave is not None:
            nflux_tensor = np.full(cropped_mask.shape + (len(new_wave),), np.nan, dtype=cropped_flux.dtype)
            iterator = np.ndindex(cropped_mask.shape)
            if progress:
                iterator = tqdm(
                    iterator,
                    total=cropped_mask.size,
                    desc="Resampling spectra"
                )
            for idx in iterator:
                if cropped_mask[idx]:
                    flux64 = 10**(cropped_flux[idx].astype(np.float64))
                    nflux64 = pyrebin.rebin_padvalue(cropped_wave, flux64, new_wave)
                    lognflux = np.log10(nflux64).astype(cropped_flux.dtype)
                    nflux_tensor[idx] = lognflux
        else:
            nflux_tensor = cropped_flux

        new_metadata = self.grid.metadata.copy()
        new_metadata['id'] = derived_model_id
        new_metadata['is_derived'] = True
        new_metadata['parent'] = parent_id
        new_metadata['wave_sampling'] = method
        
        # 实例化新的底层网格
        new_grid = SpecGrid(
            wave=new_wave,
            axes=new_axes,
            axis_names=self.grid.axis_names,
            flux_tensor=nflux_tensor,
            valid_mask=cropped_mask,
            grid_parameters=cropped_grid_pars,
            metadata=new_metadata
        )
        
        new_model = self.__class__(new_grid)

        logger.info(f"Caching derived grid to {cache_filepath}")
        new_model.grid.to_hdf5(cache_filepath)

        if not modelRegistry.exists(derived_model_id):
            modelRegistry.add_derived(
                model_id=derived_model_id, 
                parent_id=parent_id, 
                filename=cache_filename,
                # parameters={"select": select, "wavelength": wavelength}
            )
        else:
            # 应对 overwrite=True 强行覆盖的情况，确保状态为 available
            modelRegistry.update(derived_model_id, {"status": "available"})

        if alias:
            if modelRegistry.exists(alias):
                modelRegistry.update(alias, {"target": derived_model_id})
            else:
                modelRegistry.add_alias(alias, derived_model_id)

        modelRegistry.save()

        return new_model

    def get_flux(self, **kwargs):
        missing_params = set(self.grid.axis_names) - set(kwargs.keys())
        if missing_params:
            raise ValueError(
                f"Missing required grid parameters: {list(missing_params)}. "
                f"Required parameters for this model are: {list(self.grid.axis_names)}"
            )
            
        query_point = []
        bounding_slices = []
        local_axes = []
        
        for param in self.grid.axis_names:
            val = kwargs[param]
            axis_arr = self.grid.axes[param]
            min_val, max_val = np.min(axis_arr), np.max(axis_arr)
            
            if val < min_val or val > max_val:
                raise ValueError(
                    f"Parameter '{param}'={val} is outside of the grid range [{min_val}, {max_val}]."
                )
            query_point.append(val)
            
            # 定位当前参数所在的局部区间（找到紧邻的左右两个网格点）
            # 假设网格轴是升序排列的
            idx_left = np.searchsorted(axis_arr, val, side='right') - 1
            idx_left = int(np.clip(idx_left, 0, len(axis_arr) - 2))
            idx_right = idx_left + 1
            
            # 记录原生 slice 和局部坐标轴
            bounding_slices.append(slice(idx_left, idx_right + 1))
            local_axes.append(axis_arr[idx_left:idx_right + 1])
        
        bounding_slices.append(slice(None))
        
        local_flux_tensor = self.grid.flux_tensor[tuple(bounding_slices)]
        
        local_interpolator = RegularGridInterpolator(
            tuple(local_axes), 
            local_flux_tensor, 
            method='linear', 
            bounds_error=True, 
            fill_value=np.nan
        )
        
        interpolated_flux = local_interpolator([query_point])[0]
        if np.any(np.isnan(interpolated_flux)):
            raise ValueError(
                f"The requested parameters {kwargs} fall into a physical hole (invalid model region) in the grid."
            )

        return 10**interpolated_flux

    @property
    def metadata(self):
        return self.grid.metadata

    @property
    def wave(self):
        return self.grid.wave
