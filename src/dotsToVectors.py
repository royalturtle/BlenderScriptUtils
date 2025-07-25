import math
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class Vector2D:
    """Represents a single 2D vector with its properties."""
    x: float
    y: float
    length: float
    angle_deg: float

    @property
    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def angle_rad(self) -> float:
        """Get angle in radians."""
        return math.radians(self.angle_deg)

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x:.3f}, y={self.y:.3f}, length={self.length:.3f}, angle={self.angle_deg:.1f}°)"


class Vector2DList:
    """Class for creating and managing lists of 2D vectors from lengths and angles."""

    def __init__(self):
        """Initialize an empty vector list."""
        self.vectors: List[Vector2D] = []
        self._lengths: List[float] = []
        self._angles: List[float] = []

    def add_vector(self, length: float, angle_deg: float) -> None:
        """
        Add a single vector to the list.

        Args:
            length: Length/magnitude of the vector
            angle_deg: Angle in degrees (-360 to 360)
        """
        if angle_deg < -360 or angle_deg > 360:
            raise ValueError(f"Angle {angle_deg} is out of range (-360 to 360)")

        # Calculate components
        angle_rad = math.radians(angle_deg)
        x = length * math.cos(angle_rad)
        y = length * math.sin(angle_rad)

        # Create and store vector
        vector = Vector2D(x=x, y=y, length=length, angle_deg=angle_deg)
        self.vectors.append(vector)
        self._lengths.append(length)
        self._angles.append(angle_deg)

    def add_vectors_batch(self, lengths: List[float], angles: List[float]) -> None:
        """
        Add multiple vectors at once.

        Args:
            lengths: List of vector lengths
            angles: List of angles in degrees (-360 to 360)
        """
        if len(lengths) != len(angles):
            raise ValueError("Lengths and angles lists must have the same size")

        for length, angle in zip(lengths, angles):
            self.add_vector(length, angle)

    def clear(self) -> None:
        """Clear all vectors from the list."""
        self.vectors.clear()
        self._lengths.clear()
        self._angles.clear()

    def get_components(self) -> List[Tuple[float, float]]:
        """
        Get all vectors as (x, y) component tuples.

        Returns:
            List of (x, y) tuples
        """
        return [(v.x, v.y) for v in self.vectors]

    def get_numpy_array(self) -> np.ndarray:
        """
        Get all vectors as a NumPy array.

        Returns:
            NumPy array of shape (n, 2) with x, y components
        """
        if not self.vectors:
            return np.array([])
        return np.array([[v.x, v.y] for v in self.vectors])

    def get_polar_coordinates(self) -> List[Tuple[float, float]]:
        """
        Get all vectors in polar coordinates.

        Returns:
            List of (length, angle_deg) tuples
        """
        return [(v.length, v.angle_deg) for v in self.vectors]

    def scale_all(self, factor: float) -> None:
        """Scale all vectors by a factor."""
        for i, vector in enumerate(self.vectors):
            self._lengths[i] *= factor
            vector.length *= factor
            vector.x *= factor
            vector.y *= factor

    def rotate_all(self, angle_deg: float) -> None:
        """Rotate all vectors by an angle in degrees."""
        angle_rad = math.radians(angle_deg)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        for i, vector in enumerate(self.vectors):
            # Rotate using rotation matrix
            new_x = vector.x * cos_angle - vector.y * sin_angle
            new_y = vector.x * sin_angle + vector.y * cos_angle

            # Update vector
            vector.x = new_x
            vector.y = new_y
            vector.angle_deg = (vector.angle_deg + angle_deg) % 360
            self._angles[i] = vector.angle_deg

    def get_statistics(self) -> dict:
        """Get statistical information about the vectors."""
        if not self.vectors:
            return {"count": 0}

        lengths = [v.length for v in self.vectors]
        angles = [v.angle_deg for v in self.vectors]

        return {
            "count": len(self.vectors),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_angle": sum(angles) / len(angles),
            "total_x": sum(v.x for v in self.vectors),
            "total_y": sum(v.y for v in self.vectors)
        }

    def __len__(self) -> int:
        """Return the number of vectors."""
        return len(self.vectors)

    def __getitem__(self, index: int) -> Vector2D:
        """Get a vector by index."""
        return self.vectors[index]

    def __repr__(self) -> str:
        return f"Vector2DList with {len(self.vectors)} vectors"

    def print_summary(self) -> None:
        """Print a summary of all vectors."""
        print(f"\n{self}")
        print("-" * 60)
        for i, vector in enumerate(self.vectors):
            print(f"{i + 1}. {vector}")

        if self.vectors:
            stats = self.get_statistics()
            print("\nStatistics:")
            print(f"  Average length: {stats['avg_length']:.3f}")
            print(f"  Length range: [{stats['min_length']:.3f}, {stats['max_length']:.3f}]")
            print(f"  Sum of vectors: ({stats['total_x']:.3f}, {stats['total_y']:.3f})")


# Example usage and demonstrations
if __name__ == "__main__":
    # Example 1: Basic usage
    print("Example 1: Creating vectors one by one")
    vec_list = Vector2DList()

    # Add individual vectors
    vec_list.add_vector(5, 0)  # Right
    vec_list.add_vector(3, 90)  # Up
    vec_list.add_vector(4, 180)  # Left
    vec_list.add_vector(6, -45)  # Down-right

    vec_list.print_summary()

    # Example 2: Batch addition
    print("\n\nExample 2: Batch vector creation")
    vec_list2 = Vector2DList()

    lengths = [2, 3, 4, 5, 6]
    angles = [0, 72, 144, 216, 288]  # Pentagon shape
    vec_list2.add_vectors_batch(lengths, angles)

    print(f"Created {len(vec_list2)} vectors")
    print("Components:", vec_list2.get_components())

    # Example 3: Vector operations
    print("\n\nExample 3: Vector operations")
    vec_list3 = Vector2DList()
    vec_list3.add_vectors_batch([3, 3, 3, 3], [0, 90, 180, 270])

    print("Original vectors:")
    for v in vec_list3.vectors:
        print(f"  {v}")

    # Scale all vectors
    vec_list3.scale_all(2)
    print("\nAfter scaling by 2:")
    for v in vec_list3.vectors:
        print(f"  {v}")

    # Rotate all vectors
    vec_list3.rotate_all(45)
    print("\nAfter rotating by 45°:")
    for v in vec_list3.vectors:
        print(f"  {v}")

    # Example 4: Working with results
    print("\n\nExample 4: Different output formats")
    vec_list4 = Vector2DList()
    vec_list4.add_vectors_batch([1, 2, 3], [30, 60, 90])

    print("As component tuples:", vec_list4.get_components())
    print("As NumPy array:\n", vec_list4.get_numpy_array())
    print("As polar coordinates:", vec_list4.get_polar_coordinates())

    # Get statistics
    stats = vec_list4.get_statistics()
    print("\nStatistics:", stats)

    # Visualize
    print("\nVisualizing vectors...")
    vec_list.visualize()

    # Example 5: Circle of vectors
    print("\n\nExample 5: Creating a circle of vectors")
    vec_circle = Vector2DList()
    n_vectors = 12
    radius = 3

    for i in range(n_vectors):
        angle = i * 360 / n_vectors
        vec_circle.add_vector(radius, angle)

    print(f"Created {len(vec_circle)} vectors in a circle")
    vec_circle.visualize(show_labels=False)