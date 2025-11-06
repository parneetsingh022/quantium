class AffineTemperatureOperationError(TypeError):
    """Raised when an invalid arithmetic operation is performed on affine temperature units."""

    def __init__(self, operation: str):
        valid_units = ["Kelvin", "Rankine"]
        affine_units = ["Celsius", "Fahrenheit"]
        message = (
            f"The operation '{operation}' is undefined for absolute affine temperature units "
            f"({', '.join(affine_units)}). "
            f"Only ratio-scale units with zero offset ({', '.join(valid_units)}) may be used "
            f"in multiplicative or power expressions.\n\n"
            "For Celsius or Fahrenheit, use a temperature difference (ΔT) or convert to Kelvin/Rankine first."
        )
        super().__init__(message)