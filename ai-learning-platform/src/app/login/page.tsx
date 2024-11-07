"use client";

import React from "react";
import { useForm, SubmitHandler } from "react-hook-form";

type LoginFormInputs = {
  email: string;
  password: string;
};

const Login: React.FC = () => {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormInputs>();

  const onSubmit: SubmitHandler<LoginFormInputs> = (data) => {
    console.log(data);
    // You can add your login logic here, such as calling an API
  };

  return (
    <div
      style={{
        maxWidth: "400px",
        margin: "auto",
        padding: "2rem",
        textAlign: "center",
      }}
    >
      <h2>Login</h2>
      <form onSubmit={handleSubmit(onSubmit)}>
        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="email">Email:</label>
          <input
            id="email"
            type="email"
            {...register("email", { required: "Email is required" })}
            style={{ width: "100%", padding: "0.5rem", marginTop: "0.5rem" }}
          />
          {errors.email && (
            <p style={{ color: "red" }}>{errors.email.message}</p>
          )}
        </div>

        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="password">Password:</label>
          <input
            id="password"
            type="password"
            {...register("password", { required: "Password is required" })}
            style={{ width: "100%", padding: "0.5rem", marginTop: "0.5rem" }}
          />
          {errors.password && (
            <p style={{ color: "red" }}>{errors.password.message}</p>
          )}
        </div>

        <button type="submit" style={{ padding: "0.5rem 2rem" }}>
          Login
        </button>
      </form>
    </div>
  );
};

export default Login;
