from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple

from sapien.render import RenderCameraComponent
from sapien.wrapper.urdf_loader import URDFLoader as SapienURDFLoader

from mani_skill2.utils.building.actor_builder import ActorBuilder
from mani_skill2.utils.building.articulation_builder import ArticulationBuilder
from mani_skill2.utils.structs.actor import Actor
from mani_skill2.utils.structs.articulation import Articulation

if TYPE_CHECKING:
    from mani_skill2.envs.scene import ManiSkillScene


class URDFLoader(SapienURDFLoader):
    scene: ManiSkillScene
    name: str

    def parse(
        self, urdf_file, srdf_file=None, package_dir=None
    ) -> Tuple[List[ArticulationBuilder], List[ActorBuilder], List[Any]]:
        articulation_builders, actor_builders, cameras = super().parse(
            urdf_file, srdf_file, package_dir
        )
        for i, a in enumerate(articulation_builders):
            a.set_name(f"{self.name}-articulation-{i}")
        for i, b in enumerate(actor_builders):
            b.set_name(f"{self.name}-actor-{i}")
        return articulation_builders, actor_builders, cameras

    def load_file_as_articulation_builder(
        self, urdf_file, srdf_file=None, package_dir=None
    ) -> ArticulationBuilder:
        return super().load_file_as_articulation_builder(
            urdf_file, srdf_file, package_dir
        )

    def load(
        self,
        urdf_file: str,
        srdf_file=None,
        package_dir=None,
        name=None,
        scene_mask=None,
    ) -> Articulation:
        """
        Args:
            urdf_file: filename for URDL file
            srdf_file: SRDF for urdf_file. If srdf_file is None, it defaults to the ".srdf" file with the same as the urdf file
            package_dir: base directory used to resolve asset files in the URDF file. If an asset path starts with "package://", "package://" is simply removed from the file name
            name (str): name of the created articulation
        Returns:
            returns a single Articulation loaded from the URDF file. It throws an error if multiple objects exists
        """
        if name is not None:
            self.name = name
        articulation_builders, actor_builders, cameras = self.parse(
            urdf_file, srdf_file, package_dir
        )

        if len(articulation_builders) > 1 or len(actor_builders) != 0:
            raise Exception(
                "URDF contains multiple objects, call load_multiple instead"
            )

        articulations: List[Articulation] = []
        for b in articulation_builders:
            b.set_scene_mask(scene_mask)
            articulations.append(b.build())

        actors: List[Actor] = []
        for b in actor_builders:
            actors.append(b.build())

        if len(cameras) > 0:
            name2entity = dict()
            for a in articulations:
                for sapien_articulation in a._objs:
                    for l in sapien_articulation.links:
                        name2entity[l.name] = l.entity

            for a in actors:
                name2entity[a.name] = a

            for scene_idx, scene in enumerate(self.scene.sub_scenes):
                for cam in cameras:
                    cam_component = RenderCameraComponent(cam["width"], cam["height"])
                    if cam["fovx"] is not None and cam["fovy"] is not None:
                        cam_component.set_fovx(cam["fovx"], False)
                        cam_component.set_fovy(cam["fovy"], False)
                    elif cam["fovy"] is None:
                        cam_component.set_fovx(cam["fovx"], True)
                    elif cam["fovx"] is None:
                        cam_component.set_fovy(cam["fovy"], True)

                    cam_component.near = cam["near"]
                    cam_component.far = cam["far"]
                    name2entity[f"scene-{scene_idx}_{cam['reference']}"].add_component(
                        cam_component
                    )

        return articulations[0]
