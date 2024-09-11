import numpy as np
import pygame

from worldllm_envs.playground.color_generation import sample_color


class Thing:
    def __init__(self, object_descr, object_id_int, params):
        """
        Main object class.

        Parameters
        ----------
        object_descr: dict
            Dict that specify some attributes of the object that need to be created.
        object_id_int: int
            id of the object in the scene.
        params: dict
            Dict with all environment parameters.
        """

        self.object_descr = object_descr
        self.object_id_int = object_id_int
        self.params = params
        self.min_max_sizes = params["min_max_sizes"]
        self.admissible_attributes = params["admissible_attributes"]
        self.adm_rel_attributes = [
            a for a in self.admissible_attributes if "relative" in a
        ]

        self.agent_size = params["agent_size"]
        self.obj_size_update = params["obj_size_update"]
        self.get_attributes_functions = params["extract_functions"][
            "get_attributes_functions"
        ]
        self.img_path = params["img_path"]

        # initialize object attributes.
        self.object_attributes = self.object_descr.copy()
        self.object_initial_attributes = dict(
            zip(
                sorted(self.object_descr.keys()),
                [[self.object_descr[k]] for k in sorted(self.object_descr.keys())],
            )
        )
        # add relative attributes (will be filled later when all objects in the scene have been created)
        for a in self.adm_rel_attributes:
            self.object_initial_attributes[a] = []

        # Initialize physical attributes of the object that will compose its features.
        self.rgb_code = None
        self.position = None
        self.size = None
        self.type = None
        self.grasped = False
        self.scene_objects = []  # list of refs to other objects from the scene
        self.grown_once = False
        self.big = True

        # initialize values for the type, position color and size.
        self._get_type_encoding()
        self._sample_position()
        self._sample_color()
        self._sample_size()
        self.initial_rgb_code = self.rgb_code.copy()

        # rendering
        self.view = False
        self.patch = None

    def update_ref_to_scene_objects(self, scene_objects):
        """
        Add reference to other objects in the scene. Update attributes (relative ones especially).
        Parameters
        ----------
        scene_objects: list of Thing objects.

        """
        self.scene_objects = scene_objects
        for k in self.admissible_attributes:
            self._update_attribute(k)

    # Sample physical attributes of the object
    def _sample_color(self):
        """
        Sample a color for the object.

        """
        if "colors" in self.admissible_attributes:
            if "shades" in self.admissible_attributes:
                rgb_code = sample_color(
                    color=self.object_initial_attributes["colors"][0],
                    shade=self.object_initial_attributes["shades"][0],
                )
            else:
                rgb_code = sample_color(
                    color=self.object_initial_attributes["colors"][0],
                    shade=np.random.choice(["light", "dark"]),
                )
        else:
            rgb_code = np.random.uniform(-1, 1, 3)
        self._update_color(rgb_code)

    def _sample_size(self):
        """
        Sample a size for the object

        """
        size = 0.2
        self._update_size(size)

    def _sample_position(self):
        """
        Sample a position for the object.
        """
        lows = (-1, -1)
        highs = (1, 1)
        if "positions" in self.admissible_attributes:
            if self.object_initial_attributes["positions"] == "left":
                highs = (0, 1)
            elif self.object_initial_attributes["positions"] == "right":
                lows = (0, -1)
            elif self.object_initial_attributes["positions"] == "top":
                lows = (-1, 0)
            elif self.object_initial_attributes["positions"] == "bottom":
                highs = (1, 0)
            else:
                raise NotImplementedError
        ok = False
        while not ok:
            candidate_position = np.random.uniform(lows, highs)
            ok = True
            for obj in self.scene_objects:
                if obj.position is not None:
                    if (
                        np.linalg.norm(obj.position - candidate_position)
                        < self.params["epsilon_initial_pos"]
                    ):
                        ok = False
            if ok:
                self._update_position(candidate_position)

    # Update physical attributes of the object
    def _update_size(self, new_size):
        self.size = new_size
        self.size_pixels = int(self.params["ratio_size"] * self.size)
        self._update_attribute("sizes")
        if self.scene_objects:
            for obj in self.scene_objects:
                if obj is not None:
                    obj._update_attribute("relative_sizes")

    def _update_color(self, new_rgb):
        self.rgb_code = new_rgb
        self._update_attribute("colors")
        self._update_attribute("shades")
        if self.scene_objects:
            for obj in self.scene_objects:
                if obj is not None:
                    obj._update_attribute("relative_colors")
                    obj._update_attribute("relative_shades")

    def _update_position(self, new_position):
        clipped_position = np.clip(new_position, -1, 1)
        self.position = clipped_position.copy()
        self._update_attribute("positions")
        if self.scene_objects:
            for obj in self.scene_objects:
                if obj is not None:
                    obj._update_attribute("relative_positions")

    def _update_attribute(self, attribute):
        """
        Update a symbolic attribute as a function of other objects in the scene
        Parameters
        ----------
        attribute: str
            Attribute to be updated
        """
        all_attributes = [k for k in self.admissible_attributes if attribute in k]
        if len(self.scene_objects) > 0:
            object_features = [
                o.get_features() if o is not None else np.zeros(35)
                for o in self.scene_objects
            ]
            for att in all_attributes:
                self.object_attributes[att] = self.get_attributes_functions[att](
                    object_features, self.object_id_int
                )

    # Get type one hot code
    def _get_type_encoding(self):
        self.type = np.zeros([self.params["nb_types"]])
        self.type[
            self.params["attributes"]["types"].index(
                self.object_initial_attributes["types"][0]
            )
        ] = 1

    def enforce_relative_attributes(self):
        """
        When the object needs to have some relative attribute (e.g. the darkest), sample the corresponding physical property until it is.
        """
        oks = [False for _ in range(3)]
        counter = 0
        while not all(oks) and counter < 100:
            if "relative_positions" in self.admissible_attributes:
                if (
                    self.object_attributes["relative_positions"]
                    != self.object_initial_attributes["relative_positions"]
                ):
                    self._sample_position()
                else:
                    oks[0] = True
            else:
                oks[0] = True
            if "relative_shades" in self.admissible_attributes:
                if (
                    self.object_attributes["relative_shades"]
                    != self.object_initial_attributes["relative_shades"]
                ):
                    self._sample_color()
                else:
                    oks[1] = True
            else:
                oks[1] = True
            if "relative_sizes" in self.admissible_attributes:
                if (
                    self.object_attributes["relative_sizes"]
                    != self.object_initial_attributes["relative_sizes"]
                ):
                    self._sample_size()
                else:
                    oks[2] = True
            else:
                oks[2] = True
            counter += 1
        return self.assert_equal_attributes(
            self.object_initial_attributes, self.object_attributes
        )

    def assert_equal_attributes(self, att_1, att_2):
        """
        Function to check that the required attributes are included in the current attributes of the object.
        Parameters
        ----------
        att_1: dict
            Dict of attributes (current attributes).
        att_2: dict
            Dict of attributes (target attributes).

        """
        # assert that attributes 1 and included in attributes 2
        assert sorted(att_1.keys()) == sorted(att_2.keys())
        for k in att_1.keys():
            for v in att_1[k]:
                if att_2[k] is not None:
                    if v not in att_2[k]:
                        return False
        return True

    def update_state(
        self,
        agent_position,
        gripper_state,
        objects,
        object_grasped,
        action,
        rm_obj=[],
        obj_to_release=[],
    ):
        """
        Update the state of the object after interactions with the agent or other object.
        Parameters
        ----------
        agent_position: nd.array of size 2
            New agent position (2D).
        gripper_state: Boolean
            New gripper state (closed or open).
        objects: list of Thing objects
            Ref to other scene objects.
        object_grasped: Boolean
            Whether the object was grasped before.
        action: nd.array
            The action of the agent.

        """
        update_object_grasped = object_grasped

        # if the hand is close enough
        if (
            not self.grasped
            and np.linalg.norm(self.position - agent_position)
            < (self.size + self.agent_size) / 2
            and gripper_state
        ):
            self.grasped = True

        # if grasped, the object follows the hand
        if self.grasped:
            self._update_position(agent_position.copy())
        self.features = self.get_features()
        return update_object_grasped, rm_obj

    def get_features(self):
        """
        Form features of the object.
        """
        grasped_feature = np.array([1]) if self.grasped else np.array([-1])
        features = np.concatenate(
            [
                self.type,
                self.position,
                np.array([self.size]),
                self.rgb_code,
                grasped_feature,
            ]
        )
        return features

    def _color_surface(self, surface, rgb):
        arr = pygame.surfarray.pixels3d(surface)
        arr[:, :, 0] = rgb[0]
        arr[:, :, 1] = rgb[1]
        arr[:, :, 2] = rgb[2]

    def get_pixel_coordinates(self, xpos, ypos):
        return (
            (xpos + 1) / 2 * (self.params["screen_size"] * 2 / 3)
            + 1 / 6 * self.params["screen_size"]
        ).astype(np.int64), (
            (-ypos + 1) / 2 * (self.params["screen_size"] * 2 / 3)
            + 1 / 6 * self.params["screen_size"]
        ).astype(
            np.int64
        )

    def update_rendering(self, viewer):
        x, y = self.get_pixel_coordinates(self.position[0], self.position[1])
        left = int(x - self.size_pixels // 2)
        top = int(y - self.size_pixels // 2)

        color = tuple(self.rgb_code * 255)
        # pygame.draw.rect(self.icon, color, (0, 0, self.size_pixels - 1, self.size_pixels - 1), 6)
        self.icon = pygame.transform.scale(
            self.icon, (self.size_pixels, self.size_pixels)
        ).convert_alpha()
        self.surface = self.icon.copy()
        self._color_surface(self.surface, color)
        viewer.blit(self.surface, (left, top))
        # viewer.blit(self.icon, (left, top))

    def __repr__(self):
        msg = "\n\nOBJ #{}: ".format(self.object_id_int)
        for att in self.object_attributes:
            msg += "\n\t{}: {}".format(att, self.object_attributes[att])
        return msg


class LivingThings(Thing):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Animals(LivingThings):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class SmallCarnivorous(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)

    def update_state(
        self,
        hand_position,
        gripper_state,
        objects,
        object_grasped,
        action,
        rm_obj=[],
        obj_to_release=[],
    ):

        for obj in objects:

            condition = (
                obj is not None
                and obj.object_descr["categories"] == "small_herbivorous"
                and not self.grown_once
                and obj.grown_once
            )
            condition = (
                condition
                and (
                    not self.grasped
                    or self in obj_to_release
                    and len(obj_to_release) == 1
                )
                and (
                    not obj.grasped
                    or obj in obj_to_release
                    and len(obj_to_release) == 1
                )
            )

            if condition:
                # check distance
                if (
                    np.linalg.norm(obj.position - self.position)
                    < (self.size + obj.size) / 2
                    and not self.grown_once
                ):
                    self.grown_once = True
                    # check action
                    size = min(
                        self.size + self.obj_size_update,
                        self.min_max_sizes[1][1] + self.obj_size_update,
                    )
                    self._update_size(size)
                    self.grasped = False
                    rm_obj.append(obj)

        return super().update_state(
            hand_position,
            gripper_state,
            objects,
            object_grasped,
            action,
            rm_obj,
            obj_to_release,
        )


class BigCarnivorous(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)

    def update_state(
        self,
        hand_position,
        gripper_state,
        objects,
        object_grasped,
        action,
        rm_obj=[],
        obj_to_release=[],
    ):

        for i in range(len(objects) - 1):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = objects[i], objects[j]

                if obj1 == obj2:
                    continue

                condition_small = (
                    obj1 is not None
                    and obj2 is not None
                    and not self.grown_once
                    and obj1.object_descr["categories"] == "small_herbivorous"
                    and obj2.object_descr["categories"] == "small_herbivorous"
                    and obj1.grown_once
                    and obj2.grown_once
                )
                condition_small = condition_small and (
                    (
                        obj1.grasped
                        and obj2.grasped
                        and obj1 in obj_to_release
                        and obj2 in obj_to_release
                        and not self.grasped
                    )
                    or (
                        obj1.grasped
                        and self.grasped
                        and obj1 in obj_to_release
                        and self in obj_to_release
                        and not obj2.grasped
                    )
                    or (
                        obj2.grasped
                        and self.grasped
                        and obj2 in obj_to_release
                        and self in obj_to_release
                        and not obj1.grasped
                    )
                )

                if condition_small:
                    # check distance
                    if (
                        np.linalg.norm(obj1.position - self.position)
                        < (self.size + obj1.size) / 2
                        and np.linalg.norm(obj2.position - self.position)
                        < (self.size + obj2.size) / 2
                    ):
                        self.grown_once = True
                        # check action
                        size = min(
                            self.size + self.obj_size_update,
                            self.min_max_sizes[1][1] + self.obj_size_update,
                        )
                        self._update_size(size)
                        self.grasped = False
                        rm_obj.append(obj1)
                        rm_obj.append(obj2)

        if not condition_small:

            for obj in objects:

                condition_big = (
                    obj is not None
                    and obj.object_descr["categories"] == "big_herbivorous"
                    and not self.grown_once
                    and obj.grown_once
                )
                condition_big = (
                    condition_big
                    and (
                        not self.grasped
                        or self in obj_to_release
                        and len(obj_to_release) == 1
                    )
                    and (
                        not obj.grasped
                        or obj in obj_to_release
                        and len(obj_to_release) == 1
                    )
                )

                if condition_big:
                    # check distance
                    if (
                        np.linalg.norm(obj.position - self.position)
                        < (self.size + obj.size) / 2
                        and not self.grown_once
                    ):
                        self.grown_once = True
                        # check action
                        size = min(
                            self.size + self.obj_size_update,
                            self.min_max_sizes[1][1] + self.obj_size_update,
                        )
                        self._update_size(size)
                        self.grasped = False
                        rm_obj.append(obj)

        return super().update_state(
            hand_position,
            gripper_state,
            objects,
            object_grasped,
            action,
            rm_obj,
            obj_to_release,
        )


class SmallHerbivorous(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)

    def update_state(
        self,
        hand_position,
        gripper_state,
        objects,
        object_grasped,
        action,
        rm_obj=[],
        obj_to_release=[],
    ):

        for obj in objects:

            condition = (
                obj is not None
                and obj.object_descr["categories"] == "plant"
                and not self.grown_once
                and obj.grown_once
            )
            condition = (
                condition
                and (
                    not self.grasped
                    or self in obj_to_release
                    and len(obj_to_release) == 1
                )
                and (
                    not obj.grasped
                    or obj in obj_to_release
                    and len(obj_to_release) == 1
                )
            )

            if condition:
                # check distance
                if (
                    np.linalg.norm(obj.position - self.position)
                    < (self.size + obj.size) / 2
                    and not self.grown_once
                ):
                    self.grown_once = True
                    # check action
                    size = min(
                        self.size + self.obj_size_update,
                        self.min_max_sizes[1][1] + self.obj_size_update,
                    )
                    self._update_size(size)
                    self.grasped = False
                    rm_obj.append(obj)

        return super().update_state(
            hand_position,
            gripper_state,
            objects,
            object_grasped,
            action,
            rm_obj,
            obj_to_release,
        )


class BigHerbivorous(Animals):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)

    def update_state(
        self,
        hand_position,
        gripper_state,
        objects,
        object_grasped,
        action,
        rm_obj=[],
        obj_to_release=[],
    ):

        for i in range(len(objects) - 1):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = objects[i], objects[j]

                if obj1 == obj2:
                    continue

                condition = (
                    obj1 is not None
                    and obj2 is not None
                    and obj1.object_descr["categories"] == "plant"
                    and obj2.object_descr["categories"] == "plant"
                    and not self.grown_once
                    and obj1.grown_once
                    and obj2.grown_once
                )
                condition = condition and (
                    (
                        obj1.grasped
                        and obj2.grasped
                        and obj1 in obj_to_release
                        and obj2 in obj_to_release
                        and not self.grasped
                    )
                    or (
                        obj1.grasped
                        and self.grasped
                        and obj1 in obj_to_release
                        and self in obj_to_release
                        and not obj2.grasped
                    )
                    or (
                        obj2.grasped
                        and self.grasped
                        and obj2 in obj_to_release
                        and self in obj_to_release
                        and not obj1.grasped
                    )
                )

                if condition:
                    # check distance
                    if (
                        np.linalg.norm(obj1.position - self.position)
                        < (self.size + obj1.size) / 2
                        and np.linalg.norm(obj2.position - self.position)
                        < (self.size + obj2.size) / 2
                    ):
                        self.grown_once = True
                        # check action
                        size = min(
                            self.size + self.obj_size_update,
                            self.min_max_sizes[1][1] + self.obj_size_update,
                        )
                        self._update_size(size)
                        self.grasped = False
                        rm_obj.append(obj1)
                        rm_obj.append(obj2)

        return super().update_state(
            hand_position,
            gripper_state,
            objects,
            object_grasped,
            action,
            rm_obj,
            obj_to_release,
        )


class Furnitures(Thing):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Plants(LivingThings):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)

    def update_state(
        self,
        hand_position,
        gripper_state,
        objects,
        object_grasped,
        action,
        rm_obj=[],
        obj_to_release=[],
    ):
        """
        Plant objects can be grown. This function checks whether a water object is put in contact with the plant. If it is, the plant grows.

        """

        for obj in objects:
            if obj is not None and obj.object_descr["types"] == "water":
                # check distance
                if (
                    np.linalg.norm(obj.position - self.position)
                    < (self.size + obj.size) / 2
                    and not self.grown_once
                    and (
                        not self.grasped
                        or self in obj_to_release
                        and len(obj_to_release) == 1
                    )
                    and (
                        not obj.grasped
                        or obj in obj_to_release
                        and len(obj_to_release) == 1
                    )
                ):
                    self.grown_once = True
                    # check action
                    size = min(
                        self.size + self.obj_size_update,
                        self.min_max_sizes[1][1] + self.obj_size_update,
                    )
                    self._update_size(size)

                    self.grasped = False
                    rm_obj.append(obj)

        return super().update_state(
            hand_position,
            gripper_state,
            objects,
            object_grasped,
            action,
            rm_obj,
            obj_to_release,
        )


class Supplies(Thing):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


# # # # # # # # # # # # # # # # # #
# Animals
# # # # # # # # # # # # # # # # # #


# Big carnivorous


class Lion(BigCarnivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Grizzly(BigCarnivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Shark(BigCarnivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


# Small carnivorous


class Fox(SmallCarnivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Bobcat(SmallCarnivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Coyote(SmallCarnivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


# Big herbivorous


class Elephant(BigHerbivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Giraffe(BigHerbivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Rhinoceros(BigHerbivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


# Small herbivorous


class Pig(SmallHerbivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Cow(SmallHerbivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Sheep(SmallHerbivorous):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


# # # # # # # # # # # # # # # # # #
# Plants
# # # # # # # # # # # # # # # # # #


class Carrot(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Potato(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Beet(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Berry(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


class Pea(Plants):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)


# # # # # # # # # # # # # # # # # #
# Furniture
# # # # # # # # # # # # # # # # # #


class Chair(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "chair.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Sofa(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "sofa.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Sink(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "sink.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Window(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "window.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Carpet(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "carpet.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Cupboard(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "cupboard.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Desk(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "desk.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Lamp(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "lamp.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Door(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "door.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


class Table(Furnitures):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "table.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


# # # # # # # # # # # # # # # # # #
# Supply
# # # # # # # # # # # # # # # # # #


class Water(Supplies):
    def __init__(self, object_descr, object_id_int, params):
        super().__init__(object_descr, object_id_int, params)
        if params["render_mode"]:
            self.icon = pygame.image.load(self.img_path + "water.png")
            self.icon = pygame.transform.scale(
                self.icon, (self.size_pixels, self.size_pixels)
            ).convert_alpha()


obj_type_to_obj = dict(
    sheep=Sheep,
    lion=Lion,
    pig=Pig,
    cow=Cow,
    door=Door,
    chair=Chair,
    desk=Desk,
    lamp=Lamp,
    table=Table,
    cupboard=Cupboard,
    sink=Sink,
    window=Window,
    sofa=Sofa,
    carpet=Carpet,
    water=Water,
    carrot=Carrot,
    potato=Potato,
    beet=Beet,
    berry=Berry,
    pea=Pea,
    elephant=Elephant,
    giraffe=Giraffe,
    rhinoceros=Rhinoceros,
    shark=Shark,
    fox=Fox,
    bobcat=Bobcat,
    coyote=Coyote,
    grizzly=Grizzly,
)


def generate_objects(objects_descr, params):
    """
    From a list of desired objects and their attributes, generate the scene.
    Parameters
    ----------
    objects_descr: list of dict
        List of dict that describes the attributes of the desired objects.
    params: dict
        Environment parameters.

    Returns
    -------
    objs: list of Thing objects
        List of Thing objects, objects that will be included in the scene.
    """
    for o in objects_descr:
        assert (
            o["types"] in obj_type_to_obj.keys()
        ), "The object '{}' is not registered in the obj_type_to_obj dict".format(
            o["types"]
        )

    objs = [
        obj_type_to_obj[o["types"]](o, o_id_int, params)
        for o, o_id_int in zip(objects_descr, range(len(objects_descr)))
    ]

    # give each object the reference to other objects in the scene and resample their position so that they are not in contact.
    for o in objs:
        o.update_ref_to_scene_objects(objs)
        o._sample_position()

    # Enforce that relative attributes are respected (the darkest object should be the darkest physically).
    oks = [False for _ in range(len(objs))]
    while not all(oks):
        for i_o, o in enumerate(objs):
            oks[i_o] = o.enforce_relative_attributes()
    return objs


stop = 1
