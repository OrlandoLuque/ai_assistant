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
    mfa_verified: HashSet<String>,
    usage_counts: HashMap<(String, String), u32>,
    current_request_ip: Option<String>,
}

impl AccessControlManager {
    pub fn new() -> Self {
        let mut manager = Self {
            entries: Vec::new(),
            roles: HashMap::new(),
            user_roles: HashMap::new(),
            deny_list: HashSet::new(),
            mfa_verified: HashSet::new(),
            usage_counts: HashMap::new(),
            current_request_ip: None,
        };

        // Create default roles
        manager.add_role(
            Role::new("viewer")
                .with_permission(ResourceType::Conversation, Permission::Read)
                .with_permission(ResourceType::Message, Permission::Read),
        );

        manager.add_role(
            Role::new("editor")
                .inherits_from("viewer")
                .with_permission(ResourceType::Conversation, Permission::Write)
                .with_permission(ResourceType::Message, Permission::Write),
        );

        manager.add_role(
            Role::new("admin")
                .inherits_from("editor")
                .with_permission(ResourceType::Conversation, Permission::Admin)
                .with_permission(ResourceType::Settings, Permission::Admin),
        );

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
        self.deny_list
            .insert((principal.to_string(), resource_type, permission));
    }

    pub fn check_permission(
        &self,
        principal: &str,
        resource_type: ResourceType,
        permission: Permission,
        resource_id: Option<&str>,
    ) -> AccessResult {
        // Check deny list first
        if self
            .deny_list
            .contains(&(principal.to_string(), resource_type, permission))
        {
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
                    let resource_key = resource_id.unwrap_or("");
                    if let Some(reason) =
                        self.check_conditions(principal, resource_key, &entry.conditions)
                    {
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
        let mut visited = std::collections::HashSet::new();
        self.role_has_permission_inner(role_name, resource_type, permission, &mut visited)
    }

    fn role_has_permission_inner(
        &self,
        role_name: &str,
        resource_type: ResourceType,
        permission: Permission,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        // Cycle detection: skip already-visited roles
        if !visited.insert(role_name.to_string()) {
            return false;
        }
        if let Some(role) = self.roles.get(role_name) {
            // Check direct permissions
            if let Some(perms) = role.permissions.get(&resource_type) {
                if perms.contains(&permission) {
                    return true;
                }
            }

            // Check inherited roles (with cycle protection)
            for inherited in &role.inherits {
                if self.role_has_permission_inner(inherited, resource_type, permission, visited) {
                    return true;
                }
            }
        }
        false
    }

    /// Mark a principal as having completed MFA verification
    pub fn verify_mfa(&mut self, principal: &str) {
        self.mfa_verified.insert(principal.to_string());
    }

    /// Revoke MFA verification for a principal
    pub fn revoke_mfa(&mut self, principal: &str) {
        self.mfa_verified.remove(principal);
    }

    /// Check if a principal has verified MFA
    pub fn is_mfa_verified(&self, principal: &str) -> bool {
        self.mfa_verified.contains(principal)
    }

    /// Record a usage event for a principal on a resource
    pub fn record_usage(&mut self, principal: &str, resource_key: &str) {
        let key = (principal.to_string(), resource_key.to_string());
        *self.usage_counts.entry(key).or_insert(0) += 1;
    }

    /// Get usage count for a principal on a resource
    pub fn get_usage(&self, principal: &str, resource_key: &str) -> u32 {
        let key = (principal.to_string(), resource_key.to_string());
        self.usage_counts.get(&key).copied().unwrap_or(0)
    }

    /// Reset all usage counts for a principal
    pub fn reset_usage(&mut self, principal: &str) {
        self.usage_counts.retain(|(p, _), _| p != principal);
    }

    /// Set the IP address of the current request for IP-based conditions
    pub fn set_request_ip(&mut self, ip: &str) {
        self.current_request_ip = Some(ip.to_string());
    }

    /// Clear the current request IP
    pub fn clear_request_ip(&mut self) {
        self.current_request_ip = None;
    }

    fn check_conditions(
        &self,
        principal: &str,
        resource_key: &str,
        conditions: &[AccessCondition],
    ) -> Option<String> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for condition in conditions {
            match condition {
                AccessCondition::TimeRange { start, end } => {
                    if now < *start || now > *end {
                        return Some("Outside allowed time range".to_string());
                    }
                }
                AccessCondition::RequiresMfa => {
                    if !self.mfa_verified.contains(principal) {
                        return Some("MFA verification required".to_string());
                    }
                }
                AccessCondition::IpRange(cidr) => {
                    match &self.current_request_ip {
                        Some(request_ip) => {
                            if !ip_in_cidr(request_ip, cidr) {
                                return Some(format!(
                                    "IP {} not in allowed range {}",
                                    request_ip, cidr
                                ));
                            }
                        }
                        None => {
                            // Fail-closed: deny when IP is not available
                            return Some(
                                "IP not available for IpRange verification — denied by default"
                                    .to_string(),
                            );
                        }
                    }
                }
                AccessCondition::MaxUsage(max) => {
                    let key = (principal.to_string(), resource_key.to_string());
                    let current = self.usage_counts.get(&key).copied().unwrap_or(0);
                    if current >= *max {
                        return Some(format!("Usage limit exceeded ({}/{})", current, max));
                    }
                }
                AccessCondition::Custom(_name) => {
                    // Custom conditions are application-defined and cannot be evaluated here.
                    // They pass by default — the application should check them externally.
                }
            }
        }

        None
    }

    pub fn get_user_permissions(
        &self,
        principal: &str,
    ) -> HashMap<ResourceType, HashSet<Permission>> {
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

/// Check if an IPv4 address falls within a CIDR range (e.g. "192.168.1.0/24")
fn ip_in_cidr(ip: &str, cidr: &str) -> bool {
    let parts: Vec<&str> = cidr.splitn(2, '/').collect();
    if parts.len() != 2 {
        return false;
    }

    let cidr_ip = match parse_ipv4(parts[0]) {
        Some(v) => v,
        None => return false,
    };
    let prefix_len: u32 = match parts[1].parse() {
        Ok(v) if v <= 32 => v,
        _ => return false,
    };
    let request_ip = match parse_ipv4(ip) {
        Some(v) => v,
        None => return false,
    };

    if prefix_len == 0 {
        return true;
    }
    let mask = !0u32 << (32 - prefix_len);
    (request_ip & mask) == (cidr_ip & mask)
}

/// Parse an IPv4 dotted-quad string into a u32
fn parse_ipv4(s: &str) -> Option<u32> {
    let octets: Vec<&str> = s.split('.').collect();
    if octets.len() != 4 {
        return None;
    }
    let mut result: u32 = 0;
    for octet_str in &octets {
        let octet: u8 = octet_str.parse().ok()?;
        result = (result << 8) | octet as u32;
    }
    Some(result)
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
                .with_permission(Permission::Write),
        );

        assert!(acl
            .check_permission("user1", ResourceType::Conversation, Permission::Read, None)
            .is_allowed());
        assert!(!acl
            .check_permission(
                "user1",
                ResourceType::Conversation,
                Permission::Delete,
                None
            )
            .is_allowed());
    }

    #[test]
    fn test_role_based_permission() {
        let mut acl = AccessControlManager::new();

        acl.assign_role("user1", "editor");

        assert!(acl
            .check_permission("user1", ResourceType::Conversation, Permission::Read, None)
            .is_allowed());
        assert!(acl
            .check_permission("user1", ResourceType::Conversation, Permission::Write, None)
            .is_allowed());
    }

    #[test]
    fn test_deny_override() {
        let mut acl = AccessControlManager::new();

        acl.assign_role("user1", "admin");
        acl.deny("user1", ResourceType::Settings, Permission::Admin);

        assert!(!acl
            .check_permission("user1", ResourceType::Settings, Permission::Admin, None)
            .is_allowed());
    }

    #[test]
    fn test_resource_specific() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Conversation)
                .with_permission(Permission::Read)
                .for_resource("conv1"),
        );

        assert!(acl
            .check_permission(
                "user1",
                ResourceType::Conversation,
                Permission::Read,
                Some("conv1")
            )
            .is_allowed());
        assert!(!acl
            .check_permission(
                "user1",
                ResourceType::Conversation,
                Permission::Read,
                Some("conv2")
            )
            .is_allowed());
    }

    #[test]
    fn test_mfa_condition() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Settings)
                .with_permission(Permission::Admin)
                .with_condition(AccessCondition::RequiresMfa),
        );

        // Without MFA → denied
        let result = acl.check_permission("user1", ResourceType::Settings, Permission::Admin, None);
        assert!(!result.is_allowed());
        assert_eq!(
            result,
            AccessResult::Denied("MFA verification required".to_string())
        );

        // After MFA verification → allowed
        acl.verify_mfa("user1");
        assert!(acl.is_mfa_verified("user1"));
        assert!(acl
            .check_permission("user1", ResourceType::Settings, Permission::Admin, None)
            .is_allowed());

        // Revoke MFA → denied again
        acl.revoke_mfa("user1");
        assert!(!acl.is_mfa_verified("user1"));
        assert!(!acl
            .check_permission("user1", ResourceType::Settings, Permission::Admin, None)
            .is_allowed());
    }

    #[test]
    fn test_ip_range_condition() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Conversation)
                .with_permission(Permission::Read)
                .with_condition(AccessCondition::IpRange("192.168.1.0/24".to_string())),
        );

        // No IP set → fail-closed (denied)
        assert!(!acl
            .check_permission("user1", ResourceType::Conversation, Permission::Read, None)
            .is_allowed());

        // IP in range → allowed
        acl.set_request_ip("192.168.1.50");
        assert!(acl
            .check_permission("user1", ResourceType::Conversation, Permission::Read, None)
            .is_allowed());

        // IP outside range → denied
        acl.set_request_ip("10.0.0.1");
        let result =
            acl.check_permission("user1", ResourceType::Conversation, Permission::Read, None);
        assert!(!result.is_allowed());

        // Clear IP → fail-closed (denied when no IP available, H4)
        acl.clear_request_ip();
        assert!(!acl
            .check_permission("user1", ResourceType::Conversation, Permission::Read, None)
            .is_allowed());
    }

    #[test]
    fn test_max_usage_condition() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Model)
                .with_permission(Permission::Execute)
                .for_resource("gpt-4")
                .with_condition(AccessCondition::MaxUsage(3)),
        );

        // 0 usage → allowed
        assert!(acl
            .check_permission(
                "user1",
                ResourceType::Model,
                Permission::Execute,
                Some("gpt-4")
            )
            .is_allowed());

        // Record 3 usages → should be denied
        acl.record_usage("user1", "gpt-4");
        acl.record_usage("user1", "gpt-4");
        assert!(acl
            .check_permission(
                "user1",
                ResourceType::Model,
                Permission::Execute,
                Some("gpt-4")
            )
            .is_allowed());
        acl.record_usage("user1", "gpt-4");
        assert_eq!(acl.get_usage("user1", "gpt-4"), 3);
        let result = acl.check_permission(
            "user1",
            ResourceType::Model,
            Permission::Execute,
            Some("gpt-4"),
        );
        assert!(!result.is_allowed());

        // Reset usage → allowed again
        acl.reset_usage("user1");
        assert_eq!(acl.get_usage("user1", "gpt-4"), 0);
        assert!(acl
            .check_permission(
                "user1",
                ResourceType::Model,
                Permission::Execute,
                Some("gpt-4")
            )
            .is_allowed());
    }

    #[test]
    fn test_ip_in_cidr_helper() {
        // Basic /24
        assert!(ip_in_cidr("192.168.1.100", "192.168.1.0/24"));
        assert!(!ip_in_cidr("192.168.2.1", "192.168.1.0/24"));

        // /32 = exact match
        assert!(ip_in_cidr("10.0.0.1", "10.0.0.1/32"));
        assert!(!ip_in_cidr("10.0.0.2", "10.0.0.1/32"));

        // /0 = match all
        assert!(ip_in_cidr("1.2.3.4", "0.0.0.0/0"));

        // /16
        assert!(ip_in_cidr("172.16.255.1", "172.16.0.0/16"));
        assert!(!ip_in_cidr("172.17.0.1", "172.16.0.0/16"));

        // Invalid inputs
        assert!(!ip_in_cidr("not-an-ip", "192.168.1.0/24"));
        assert!(!ip_in_cidr("192.168.1.1", "bad-cidr"));
        assert!(!ip_in_cidr("192.168.1.1", "192.168.1.0/33"));
    }

    #[test]
    fn test_custom_condition_passes() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Plugin)
                .with_permission(Permission::Execute)
                .with_condition(AccessCondition::Custom("require_license".to_string())),
        );

        // Custom conditions pass by default (application must check externally)
        assert!(acl
            .check_permission("user1", ResourceType::Plugin, Permission::Execute, None)
            .is_allowed());
    }

    #[test]
    fn test_multiple_conditions() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Settings)
                .with_permission(Permission::Admin)
                .with_condition(AccessCondition::RequiresMfa)
                .with_condition(AccessCondition::IpRange("10.0.0.0/8".to_string())),
        );

        // Neither condition met (no IP = permissive, but MFA required)
        assert!(!acl
            .check_permission("user1", ResourceType::Settings, Permission::Admin, None)
            .is_allowed());

        // MFA verified but wrong IP
        acl.verify_mfa("user1");
        acl.set_request_ip("192.168.1.1");
        assert!(!acl
            .check_permission("user1", ResourceType::Settings, Permission::Admin, None)
            .is_allowed());

        // MFA verified and correct IP
        acl.set_request_ip("10.5.3.1");
        assert!(acl
            .check_permission("user1", ResourceType::Settings, Permission::Admin, None)
            .is_allowed());
    }

    #[test]
    fn test_with_permissions_batch() {
        let mut acl = AccessControlManager::new();

        acl.add_entry(
            AccessControlEntry::new("user1", ResourceType::Memory).with_permissions(&[
                Permission::Read,
                Permission::Write,
                Permission::Delete,
            ]),
        );

        assert!(acl
            .check_permission("user1", ResourceType::Memory, Permission::Read, None)
            .is_allowed());
        assert!(acl
            .check_permission("user1", ResourceType::Memory, Permission::Write, None)
            .is_allowed());
        assert!(acl
            .check_permission("user1", ResourceType::Memory, Permission::Delete, None)
            .is_allowed());
        assert!(!acl
            .check_permission("user1", ResourceType::Memory, Permission::Admin, None)
            .is_allowed());
    }
}
