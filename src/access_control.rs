//! Access control lists
//!
//! Manage access permissions for resources.

use std::collections::{HashMap, HashSet};

/// Permission types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Permission {
    Read,
    Write,
    Delete,
    Execute,
    Admin,
    Share,
}

/// Resource types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Conversation,
    Message,
    Memory,
    Model,
    Plugin,
    Settings,
}

/// Access control entry
#[derive(Debug, Clone)]
pub struct AccessControlEntry {
    pub principal: String,
    pub permissions: HashSet<Permission>,
    pub resource_type: ResourceType,
    pub resource_id: Option<String>,
    pub conditions: Vec<AccessCondition>,
}

/// Access condition
#[derive(Debug, Clone)]
pub enum AccessCondition {
    TimeRange { start: u64, end: u64 },
    IpRange(String),
    MaxUsage(u32),
    RequiresMfa,
    Custom(String),
}

impl AccessControlEntry {
    pub fn new(principal: &str, resource_type: ResourceType) -> Self {
        Self {
            principal: principal.to_string(),
            permissions: HashSet::new(),
            resource_type,
            resource_id: None,
            conditions: Vec::new(),
        }
    }

    pub fn with_permission(mut self, permission: Permission) -> Self {
        self.permissions.insert(permission);
        self
    }

    pub fn with_permissions(mut self, permissions: &[Permission]) -> Self {
        for p in permissions {
            self.permissions.insert(*p);
        }
        self
    }

    pub fn for_resource(mut self, resource_id: &str) -> Self {
        self.resource_id = Some(resource_id.to_string());
        self
    }

    pub fn with_condition(mut self, condition: AccessCondition) -> Self {
        self.conditions.push(condition);
        self
    }
}

/// Role definition
#[derive(Debug, Clone)]
pub struct Role {
    pub name: String,
    pub permissions: HashMap<ResourceType, HashSet<Permission>>,
    pub inherits: Vec<String>,
}

impl Role {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            permissions: HashMap::new(),
            inherits: Vec::new(),
        }
    }

    pub fn with_permission(mut self, resource_type: ResourceType, permission: Permission) -> Self {
        self.permissions
            .entry(resource_type)
            .or_default()
            .insert(permission);
        self
    }

    pub fn inherits_from(mut self, role_name: &str) -> Self {
        self.inherits.push(role_name.to_string());
        self
    }
}

/// Access control list manager
pub struct AccessControlManager {
    entries: Vec<AccessControlEntry>,
    roles: HashMap<String, Role>,
    user_roles: HashMap<String, Vec<String>>,
    deny_list: HashSet<(String, ResourceType, Permission)>,
}

impl AccessControlManager {
    pub fn new() -> Self {
        let mut manager = Self {
            entries: Vec::new(),
            roles: HashMap::new(),
            user_roles: HashMap::new(),
            deny_list: HashSet::new(),
        };

        // Create default roles
        manager.add_role(Role::new("viewer")
            .with_permission(ResourceType::Conversation, Permission::Read)
            .with_permission(ResourceType::Message, Permission::Read));

        manager.add_role(Role::new("editor")
            .inherits_from("viewer")
            .with_permission(ResourceType::Conversation, Permission::Write)
            .with_permission(ResourceType::Message, Permission::Write));

        manager.add_role(Role::new("admin")
            .inherits_from("editor")
            .with_permission(ResourceType::Conversation, Permission::Admin)
            .with_permission(ResourceType::Settings, Permission::Admin));

        manager
    }

    pub fn add_entry(&mut self, entry: AccessControlEntry) {
        self.entries.push(entry);
    }

    pub fn add_role(&mut self, role: Role) {
        self.roles.insert(role.name.clone(), role);
    }

    pub fn assign_role(&mut self, principal: &str, role_name: &str) {
        self.user_roles
            .entry(principal.to_string())
            .or_default()
            .push(role_name.to_string());
    }

    pub fn revoke_role(&mut self, principal: &str, role_name: &str) {
        if let Some(roles) = self.user_roles.get_mut(principal) {
            roles.retain(|r| r != role_name);
        }
    }

    pub fn deny(&mut self, principal: &str, resource_type: ResourceType, permission: Permission) {
        self.deny_list.insert((principal.to_string(), resource_type, permission));
    }

    pub fn check_permission(
        &self,
        principal: &str,
        resource_type: ResourceType,
        permission: Permission,
        resource_id: Option<&str>,
    ) -> AccessResult {
        // Check deny list first
        if self.deny_list.contains(&(principal.to_string(), resource_type, permission)) {
            return AccessResult::Denied("Explicitly denied".to_string());
        }

        // Check direct entries
        for entry in &self.entries {
            if entry.principal == principal && entry.resource_type == resource_type {
                if let Some(ref rid) = entry.resource_id {
                    if resource_id != Some(rid.as_str()) {
                        continue;
                    }
                }

                if entry.permissions.contains(&permission) {
                    // Check conditions
                    if let Some(reason) = self.check_conditions(&entry.conditions) {
                        return AccessResult::Denied(reason);
                    }
                    return AccessResult::Allowed;
                }
            }
        }

        // Check role-based permissions
        if let Some(user_roles) = self.user_roles.get(principal) {
            for role_name in user_roles {
                if self.role_has_permission(role_name, resource_type, permission) {
                    return AccessResult::Allowed;
                }
            }
        }

        AccessResult::Denied("No matching permission".to_string())
    }

    fn role_has_permission(
        &self,
        role_name: &str,
        resource_type: ResourceType,
        permission: Permission,
    ) -> bool {
        if let Some(role) = self.roles.get(role_name) {
            // Check direct permissions
            if let Some(perms) = role.permissions.get(&resource_type) {
                if perms.contains(&permission) {
                    return true;
                }
            }

            // Check inherited roles
            for inherited in &role.inherits {
                if self.role_has_permission(inherited, resource_type, permission) {
                    return true;
                }
            }
        }
        false
    }

    fn check_conditions(&self, conditions: &[AccessCondition]) -> Option<String> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for condition in conditions {
            match condition {
                AccessCondition::TimeRange { start, end } => {
                    if now < *start || now > *end {
                        return Some("Outside allowed time range".to_string());
                    }
                }
                AccessCondition::RequiresMfa => {
                    // In real implementation, check MFA status
                    return Some("MFA required".to_string());
                }
                _ => {}
            }
        }

        None
    }

    pub fn get_user_permissions(&self, principal: &str) -> HashMap<ResourceType, HashSet<Permission>> {
        let mut all_perms: HashMap<ResourceType, HashSet<Permission>> = HashMap::new();

        // Collect from direct entries
        for entry in &self.entries {
            if entry.principal == principal {
                all_perms
                    .entry(entry.resource_type)
                    .or_default()
                    .extend(entry.permissions.iter());
            }
        }

        // Collect from roles
        if let Some(user_roles) = self.user_roles.get(principal) {
            for role_name in user_roles {
                self.collect_role_permissions(role_name, &mut all_perms);
            }
        }

        all_perms
    }

    fn collect_role_permissions(
        &self,
        role_name: &str,
        perms: &mut HashMap<ResourceType, HashSet<Permission>>,
    ) {
        if let Some(role) = self.roles.get(role_name) {
            for (resource_type, role_perms) in &role.permissions {
                perms
                    .entry(*resource_type)
                    .or_default()
                    .extend(role_perms.iter());
            }

            for inherited in &role.inherits {
                self.collect_role_permissions(inherited, perms);
            }
        }
    }

    pub fn list_principals_with_access(
        &self,
        resource_type: ResourceType,
        permission: Permission,
    ) -> Vec<String> {
        let mut principals = HashSet::new();

        for entry in &self.entries {
            if entry.resource_type == resource_type && entry.permissions.contains(&permission) {
                principals.insert(entry.principal.clone());
            }
        }

        for (principal, roles) in &self.user_roles {
            for role_name in roles {
                if self.role_has_permission(role_name, resource_type, permission) {
                    principals.insert(principal.clone());
                    break;
                }
            }
        }

        principals.into_iter().collect()
    }
}

impl Default for AccessControlManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Access check result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessResult {
    Allowed,
    Denied(String),
}

impl AccessResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_permission() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Conversation)
                .with_permission(Permission::Read)
                .with_permission(Permission::Write)
        );

        assert!(acl.check_permission("user1", ResourceType::Conversation, Permission::Read, None).is_allowed());
        assert!(!acl.check_permission("user1", ResourceType::Conversation, Permission::Delete, None).is_allowed());
    }

    #[test]
    fn test_role_based_permission() {
        let mut acl = AccessControlManager::new();

        acl.assign_role("user1", "editor");

        assert!(acl.check_permission("user1", ResourceType::Conversation, Permission::Read, None).is_allowed());
        assert!(acl.check_permission("user1", ResourceType::Conversation, Permission::Write, None).is_allowed());
    }

    #[test]
    fn test_deny_override() {
        let mut acl = AccessControlManager::new();

        acl.assign_role("user1", "admin");
        acl.deny("user1", ResourceType::Settings, Permission::Admin);

        assert!(!acl.check_permission("user1", ResourceType::Settings, Permission::Admin, None).is_allowed());
    }

    #[test]
    fn test_resource_specific() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Conversation)
                .with_permission(Permission::Read)
                .for_resource("conv1")
        );

        assert!(acl.check_permission("user1", ResourceType::Conversation, Permission::Read, Some("conv1")).is_allowed());
        assert!(!acl.check_permission("user1", ResourceType::Conversation, Permission::Read, Some("conv2")).is_allowed());
    }
}
