"""
IITP Digital Twin - USD scene manager.

This module is the Omniverse/USD-facing layer of the insulin-pump digital
twin extension.

Responsibilities:
    - Access the currently open USD stage.
    - Resolve named pump/CGM prims from the existing scene.
    - Keep configurable USD paths for the patient, pump, screen, buttons,
      insulin tube, and CGM.
    - Read/write transform information where appropriate.
    - Provide safe helper methods for visualizer modules.

It intentionally does NOT:
    - run SimGlucose
    - run DT1 or DT2
    - make safety decisions
    - send WebSocket messages
    - directly command insulin delivery

The existing USD scene remains the source of geometry. This class should
operate on the scene the user has already built rather than recreating the
pump or CGM every frame.

Typical hierarchy:

    /World
        /Patient
        /InsulinPump
            /Pump_Body
            /Pump_Screen
            /Button_1
            /Button_2
            /Button_3
            /InsulinTube
        /CGM

The actual paths can differ. Configure them through USDSceneManager's
paths dictionary when integrating with the user's scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import carb
import omni.usd

from pxr import Gf, Sdf, Usd, UsdGeom


# ----------------------------------------------------------------------
# Scene path configuration
# ----------------------------------------------------------------------

@dataclass
class USDScenePaths:
    """
    Configurable paths for the existing insulin-pump digital-twin scene.

    These defaults are intentionally simple. The real scene may use paths
    such as /World/Patient/InsulinPump or generated asset paths; call
    resolve_prim_path() / set_path() during integration if needed.
    """

    world: str = "/World"

    patient: str = "/World/Patient"

    insulin_pump: str = "/World/InsulinPump"
    pump_body: str = "/World/InsulinPump/Pump_Body"
    pump_screen: str = "/World/InsulinPump/Pump_Screen"

    button_1: str = "/World/InsulinPump/Button_1"
    button_2: str = "/World/InsulinPump/Button_2"
    button_3: str = "/World/InsulinPump/Button_3"

    insulin_tube: str = "/World/InsulinPump/InsulinTube"

    cgm: str = "/World/CGM"

    def as_dict(self) -> Dict[str, str]:
        """Return all configured scene paths as a dictionary."""
        return {
            "world": self.world,
            "patient": self.patient,
            "insulin_pump": self.insulin_pump,
            "pump_body": self.pump_body,
            "pump_screen": self.pump_screen,
            "button_1": self.button_1,
            "button_2": self.button_2,
            "button_3": self.button_3,
            "insulin_tube": self.insulin_tube,
            "cgm": self.cgm,
        }


# ----------------------------------------------------------------------
# USD scene manager
# ----------------------------------------------------------------------

class USDSceneManager:
    """
    Central USD-stage helper for the IITP Omniverse digital twin.

    Visualizer modules should use this class instead of repeatedly calling
    omni.usd.get_context().get_stage() themselves.
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        paths: Optional[USDScenePaths] = None,
        context_name: str = "",
    ):
        self.context_name = context_name
        self.paths = paths or USDScenePaths()

        self._stage = stage

        carb.log_info("[IITP DT] USD scene manager initialized")

    # ------------------------------------------------------------------
    # Stage access
    # ------------------------------------------------------------------

    def get_stage(self) -> Optional[Usd.Stage]:
        """
        Return the configured stage.

        If no explicit stage was supplied, use the active Omniverse USD
        context.
        """
        if self._stage is not None:
            return self._stage

        try:
            context = omni.usd.get_context(self.context_name)
            self._stage = context.get_stage()
        except Exception as exc:
            carb.log_error(
                f"[IITP DT] Failed to access USD stage: {exc}"
            )
            return None

        return self._stage

    def refresh_stage(self) -> Optional[Usd.Stage]:
        """Re-acquire the active USD stage."""
        self._stage = None
        return self.get_stage()

    def set_stage(self, stage: Optional[Usd.Stage]) -> None:
        """Explicitly set the stage used by this manager."""
        self._stage = stage

    @property
    def has_stage(self) -> bool:
        """Return True if a valid USD stage is currently available."""
        stage = self.get_stage()
        return stage is not None

    # ------------------------------------------------------------------
    # Prim access
    # ------------------------------------------------------------------

    def get_prim(
        self,
        path: str,
    ) -> Optional[Usd.Prim]:
        """Return a USD prim by path, or None if it does not exist."""
        stage = self.get_stage()

        if stage is None:
            carb.log_warn(
                "[IITP DT] Cannot resolve prim: no USD stage"
            )
            return None

        if not path:
            return None

        try:
            prim = stage.GetPrimAtPath(Sdf.Path(path))

            if not prim or not prim.IsValid():
                return None

            return prim

        except Exception as exc:
            carb.log_warn(
                f"[IITP DT] Failed to resolve prim {path}: {exc}"
            )
            return None

    def prim_exists(self, path: str) -> bool:
        """Return True if the supplied USD path resolves to a valid prim."""
        return self.get_prim(path) is not None

    def get_named_prim(
        self,
        name: str,
    ) -> Optional[Usd.Prim]:
        """
        Resolve a configured named prim.

        Example:
            get_named_prim("pump_screen")
            get_named_prim("cgm")
        """
        path = self.paths.as_dict().get(name)

        if path is None:
            carb.log_warn(
                f"[IITP DT] Unknown configured prim name: {name}"
            )
            return None

        return self.get_prim(path)

    # ------------------------------------------------------------------
    # Path discovery
    # ------------------------------------------------------------------

    def find_first_existing_path(
        self,
        candidate_paths: Iterable[str],
    ) -> Optional[str]:
        """Return the first candidate path that exists on the stage."""
        for path in candidate_paths:
            if self.prim_exists(path):
                return path

        return None

    def resolve_prim_path(
        self,
        candidates: Sequence[str],
    ) -> Optional[str]:
        """
        Resolve a prim from a list of candidate paths.

        This is useful because imported USD assets often have different
        hierarchy names between versions.
        """
        return self.find_first_existing_path(candidates)

    def find_prims_by_name(
        self,
        name: str,
        root_path: str = "/World",
    ) -> List[str]:
        """
        Find prim paths below root_path whose prim name matches `name`.
        """
        stage = self.get_stage()

        if stage is None:
            return []

        root = stage.GetPrimAtPath(Sdf.Path(root_path))

        if not root or not root.IsValid():
            return []

        matches: List[str] = []

        for prim in Usd.PrimRange(root):
            if prim.GetName() == name:
                matches.append(prim.GetPath().pathString)

        return matches

    def find_prims_containing(
        self,
        text: str,
        root_path: str = "/World",
    ) -> List[str]:
        """
        Find prims whose names contain the requested text.

        Case-insensitive convenience function for scene integration.
        """
        stage = self.get_stage()

        if stage is None:
            return []

        root = stage.GetPrimAtPath(Sdf.Path(root_path))

        if not root or not root.IsValid():
            return []

        needle = str(text).lower()
        matches: List[str] = []

        for prim in Usd.PrimRange(root):
            if needle in prim.GetName().lower():
                matches.append(prim.GetPath().pathString)

        return matches

    # ------------------------------------------------------------------
    # Xform helpers
    # ------------------------------------------------------------------

    def get_xformable(
        self,
        path: str,
    ) -> Optional[UsdGeom.Xformable]:
        """Return UsdGeom.Xformable for a prim if supported."""
        prim = self.get_prim(path)

        if prim is None:
            return None

        try:
            xformable = UsdGeom.Xformable(prim)

            if not xformable:
                return None

            return xformable

        except Exception as exc:
            carb.log_warn(
                f"[IITP DT] Prim is not Xformable: {path}: {exc}"
            )
            return None

    def get_local_transform(
        self,
        path: str,
    ) -> Optional[Gf.Matrix4d]:
        """
        Return the local transform matrix for a prim.
        """
        xformable = self.get_xformable(path)

        if xformable is None:
            return None

        try:
            return xformable.GetLocalTransformation()[0]
        except Exception as exc:
            carb.log_warn(
                f"[IITP DT] Failed to read local transform {path}: {exc}"
            )
            return None

    def get_world_transform(
        self,
        path: str,
    ) -> Optional[Gf.Matrix4d]:
        """
        Return the local-to-world transform matrix.

        This follows the standard USD Xformable approach used by Omniverse
        APIs for reading world transforms.
        """
        xformable = self.get_xformable(path)

        if xformable is None:
            return None

        try:
            return xformable.ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
        except Exception as exc:
            carb.log_warn(
                f"[IITP DT] Failed to read world transform {path}: {exc}"
            )
            return None

    def get_world_position(
        self,
        path: str,
    ) -> Optional[Gf.Vec3d]:
        """Return the world-space translation of a prim."""
        matrix = self.get_world_transform(path)

        if matrix is None:
            return None

        try:
            return matrix.ExtractTranslation()
        except Exception as exc:
            carb.log_warn(
                f"[IITP DT] Failed to extract position {path}: {exc}"
            )
            return None

    def set_translation(
        self,
        path: str,
        translation: Sequence[float],
    ) -> bool:
        """
        Set the local translation of a prim.

        This should normally be used on an editable Xform parent, not on
        referenced mesh internals.
        """
        xformable = self.get_xformable(path)

        if xformable is None:
            return False

        if len(translation) != 3:
            carb.log_warn(
                f"[IITP DT] Translation must contain 3 values: {translation}"
            )
            return False

        try:
            ops = xformable.GetOrderedXformOps()
            translate_op = None

            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break

            if translate_op is None:
                translate_op = xformable.AddTranslateOp()

            translate_op.Set(
                Gf.Vec3d(
                    float(translation[0]),
                    float(translation[1]),
                    float(translation[2]),
                )
            )

            return True

        except Exception as exc:
            carb.log_error(
                f"[IITP DT] Failed to set translation {path}: {exc}"
            )
            return False

    def set_scale(
        self,
        path: str,
        scale: Sequence[float],
    ) -> bool:
        """Set the local scale of an Xformable prim."""
        xformable = self.get_xformable(path)

        if xformable is None:
            return False

        if len(scale) != 3:
            return False

        try:
            ops = xformable.GetOrderedXformOps()
            scale_op = None

            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break

            if scale_op is None:
                scale_op = xformable.AddScaleOp()

            scale_op.Set(
                Gf.Vec3f(
                    float(scale[0]),
                    float(scale[1]),
                    float(scale[2]),
                )
            )

            return True

        except Exception as exc:
            carb.log_error(
                f"[IITP DT] Failed to set scale {path}: {exc}"
            )
            return False

    # ------------------------------------------------------------------
    # Pump / CGM convenience helpers
    # ------------------------------------------------------------------

    def get_pump_paths(self) -> Dict[str, str]:
        """Return configured insulin-pump paths."""
        values = self.paths.as_dict()

        return {
            key: value
            for key, value in values.items()
            if key.startswith("pump_")
            or key in {"insulin_pump", "insulin_tube"}
        }

    def get_cgm_path(self) -> str:
        """Return the configured CGM path."""
        return self.paths.cgm

    def get_pump_prim(
        self,
        component: str = "insulin_pump",
    ) -> Optional[Usd.Prim]:
        """Return a configured pump component prim."""
        return self.get_named_prim(component)

    def get_cgm_prim(self) -> Optional[Usd.Prim]:
        """Return the configured CGM prim."""
        return self.get_named_prim("cgm")

    # ------------------------------------------------------------------
    # Scene diagnostics
    # ------------------------------------------------------------------

    def validate_required_prims(
        self,
        required: Optional[Sequence[str]] = None,
    ) -> Dict[str, bool]:
        """
        Check whether required configured components exist.

        Returns:
            {
                "insulin_pump": True,
                "pump_screen": True,
                ...
            }
        """
        if required is None:
            required = [
                "insulin_pump",
                "pump_body",
                "pump_screen",
                "button_1",
                "button_2",
                "button_3",
                "cgm",
            ]

        result: Dict[str, bool] = {}

        for name in required:
            result[name] = self.get_named_prim(name) is not None

        return result

    def log_scene_diagnostics(
        self,
        required: Optional[Sequence[str]] = None,
    ) -> Dict[str, bool]:
        """Validate and log the configured scene components."""
        result = self.validate_required_prims(required)

        for name, exists in result.items():
            if exists:
                carb.log_info(
                    f"[IITP DT] USD prim OK: {name} -> "
                    f"{self.paths.as_dict().get(name)}"
                )
            else:
                carb.log_warn(
                    f"[IITP DT] USD prim missing: {name} -> "
                    f"{self.paths.as_dict().get(name)}"
                )

        return result

    # ------------------------------------------------------------------
    # Path updates
    # ------------------------------------------------------------------

    def set_path(
        self,
        name: str,
        path: str,
    ) -> bool:
        """
        Change one configured scene path at runtime.

        Useful when the user's imported USD hierarchy differs from the
        default hierarchy.
        """
        if not hasattr(self.paths, name):
            carb.log_warn(
                f"[IITP DT] Cannot configure unknown scene path: {name}"
            )
            return False

        setattr(self.paths, name, str(path))

        carb.log_info(
            f"[IITP DT] Scene path updated: {name} -> {path}"
        )

        return True


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_usd_scene_manager(
    paths: Optional[USDScenePaths] = None,
    context_name: str = "",
) -> USDSceneManager:
    """Create a USDSceneManager for the active Omniverse stage."""
    return USDSceneManager(
        paths=paths,
        context_name=context_name,
    )
