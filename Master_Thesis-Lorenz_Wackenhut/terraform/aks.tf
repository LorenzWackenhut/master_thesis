// k8s cluster
resource "azurerm_kubernetes_cluster" "nc_aks" {
  name                = var.aks_cluster_name
  location            = var.resource_group_location
  resource_group_name = var.resource_group_name
  dns_prefix          = var.dns_prefix

  // az aks get-versions --location northeurope --output table
  kubernetes_version = "1.20.5"

  linux_profile {
    admin_username = "ubuntu"
    ssh_key {
      key_data = file(var.ssh_public_key)
    }
  }

  default_node_pool {
    enable_auto_scaling = true
    min_count           = var.instance_min
    max_count           = var.instance_max
    name                = "agentpool"
    node_count          = var.node_count
    vm_size             = var.instance_type
    vnet_subnet_id      = var.nc_aks_vnet_subnet_id
  }

  depends_on = [
  azurerm_container_registry.nc_acr]

  service_principal {
    client_id = var.client_id
    // client_id_aks
    client_secret = var.client_secret
    // client_secret_aks
  }
  tags = {
    company    = var.tag_company
    department = var.tag_department
    owner      = var.tag_owner
    project    = var.tag_project
    sap        = var.tag_sap
    status     = var.tag_status
    enviroment = var.tag_enviroment
  }
}


resource "azurerm_container_registry" "nc_acr" {
  name                = var.nc_acr_name
  resource_group_name = var.resource_group_name
  location            = var.resource_group_location
  sku                 = "Basic"
  admin_enabled       = false
}


resource "azurerm_storage_account" "nc_aks_storage_account" {
  name                     = var.nc_aks_storage_account_name
  resource_group_name      = var.resource_group_name
  location                 = var.resource_group_location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = "true"

  tags = {
    company    = var.tag_company
    department = var.tag_department
    owner      = var.tag_owner
    project    = var.tag_project
    sap        = var.tag_sap
    status     = var.tag_status
    enviroment = var.tag_enviroment
  }
}


resource "azurerm_storage_container" "nc_aks_storage_container" {
  name                  = var.nc_aks_storage_container_name
  storage_account_name  = azurerm_storage_account.nc_aks_storage_account.name
#  container_access_type = "blob"
  container_access_type    = "private"


  depends_on = [  azurerm_storage_account.nc_aks_storage_account]
}
