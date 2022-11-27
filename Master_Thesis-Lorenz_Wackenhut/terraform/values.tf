
// Ressource group
variable resource_group_name {
  default = "REDACTED" // link this to declaration in main.tf
}

variable resource_group_location {
  default = "westeurope"
}

variable environment {
  default = "dev"
}


// Tags
variable tag_company {
  default = "REDACTED"
}

variable tag_department {
  default = ""
}

variable tag_owner {
  default = "LW"
}

variable tag_project {
  default = ""
}

variable tag_sap {
  default = ""
}

variable tag_status {
  default = ""
}

variable tag_enviroment {
  default = ""
}






// Aks
variable aks_cluster_name {
  default = "REDACTED"
}

variable "dns_prefix" {
  default = "REDACTED"
}

variable nc_aks_storage_account_name {
  default = "REDACTED"
}

variable nc_acr_name {
  default = "REDACTED"
}

variable nc_aks_storage_container_name {
  default = "jars"
}

variable node_count {
  default = 1
}

variable "ssh_public_key" {
  default = "~/.ssh/id_rsa.pub"
}

variable "instance_type" {
# Liste der verf�gbaren VM typen abfragen: az vm list-sizes --location westeurope
#  default = "Standard_D4a_v4"
#  default = "Standard_E8a_v4"
  default = "Standard_D2as_v4"
}

variable "instance_min" {
  default = "2"
}

variable "instance_max" {
  default = "2"
}

variable "enable_auto_scaling" {
  default = "false"
}


// Authentication
variable "client_id" {
  type        = string
  default     = "..."
  description = "The client_id to use for authentication, taken from user-specific environment variables TF_VAR_client_id."

  // validation {
  //   condition     = length(var.client_id) == 36
  //   error_message = "The client_id value must be a string of 36 characters."
  // }
}

variable "client_secret" {
  type        = string
  default     = "..."
  description = "The client_secret to use for authentication, taken from user-specific environment variables TF_VAR_client_id."

  // validation {
  //   condition     = length(var.client_secret) == 32
  //   error_message = "The client_secret value must be a string of 32 characters."
  // }
}

variable "subscription_id" {
  type        = string
  default     = "..."
  description = "The subscription_id to use for authentication, taken from user-specific environment variables TF_VAR_subscription_id."

  // validation {
  //   condition     = length(var.subscription_id) == 36
  //   error_message = "The subscription_id value must be a string of 36 characters."
  // }
}

variable "tenant_id" {
  type        = string
  default     = "..."
  description = "The tenant_id to use for authentication, taken from user-specific environment variables TF_VAR_tenant_id."

  // validation {
  //   condition     = length(var.tenant_id) == 36
  //   error_message = "The tenant_id value must be a string of 36 characters."
  // }
}

variable "backend_key" {
  type    = string
  default = "..."

  // validation {
  //   condition     = length(var.backend_key) == 88
  //   error_message = "The tenant_id value must be a string of 88 characters."
  // }
}

variable nc_aks_vnet_subnet_id {
  default = "REDACTED"
}
