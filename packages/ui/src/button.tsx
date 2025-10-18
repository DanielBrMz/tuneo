import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
}

export const Button: React.FC<ButtonProps> = ({ 
  children, 
  variant = "primary",
  className = "",
  ...props 
}) => {
  const baseStyles = "px-4 py-2 rounded-lg font-semibold transition-colors";
  const variantStyles = {
    primary: "bg-purple-600 hover:bg-purple-700 text-white",
    secondary: "bg-gray-200 hover:bg-gray-300 text-gray-800",
  };

  return (
    <button 
      className={`${baseStyles} ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};