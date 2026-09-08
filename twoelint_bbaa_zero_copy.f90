! /twoelint_bbaa_zero_copy_v3.f90
!
! Zero-copy Fortran kernel for calc_BBAA (overlap-only path).
! This implementation matches the Python logic in:
!   twoelint_calc_ov_binary_2_corr.py : calc_BBAA and helpers
!
module twoelint_bbaa_mod
  use twoelint_core_mod, only: u16_from_i16

  ! iso_fortran_env removed for f2py wrapper compatibility
  implicit none
  private
  public :: bbaa_accum_all

contains

  pure double precision function get2(a, nc, r0, c0) result(x)
    double precision, intent(in) :: a(*)
    integer, intent(in) :: nc
    integer, intent(in) :: r0, c0
    x = a(int(r0, 8) * int(nc, 8) + int(c0, 8) + 1_8)
  end function get2

  subroutine split_ab(i, j, k, l, nbas_a, a0, a1, b0, b1, ok)
    integer, intent(in) :: i, j, k, l, nbas_a
    integer, intent(out) :: a0, a1, b0, b1
    logical, intent(out) :: ok
    integer :: na, nb
    integer :: aa(2), bb(2)

    na = 0
    nb = 0

    call push_idx(i)
    call push_idx(j)
    call push_idx(k)
    call push_idx(l)

    ok = (na == 2 .and. nb == 2)
    if (.not. ok) then
      a0 = 0; a1 = 0; b0 = 0; b1 = 0
      return
    end if
    a0 = aa(1); a1 = aa(2)
    b0 = bb(1); b1 = bb(2)

  contains
    subroutine push_idx(x)
      integer, intent(in) :: x
      if (x <= nbas_a) then
        na = na + 1
        if (na <= 2) aa(na) = x - 1
      else
        nb = nb + 1
        if (nb <= 2) bb(nb) = x - 1 - nbas_a
      end if
    end subroutine push_idx
  end subroutine split_ab

  subroutine bbaa_accum_all( &
      pos2, pos3, pos4, a, b, c, d, v, nbas_a, &
      sc_a_ba, sc_a_ba_nc, cs_a_ab, cs_a_ab_nc, sc_b_ab, sc_b_ab_nc, cs_b_ba, cs_b_ba_nc, &
      cs_ea_a, cs_ea_a_nc, sc_ip_a, sc_ip_a_nc, sc_ip_b, sc_ip_b_nc, cs_ea_b, cs_ea_b_nc, &
      sc_a_ba_pa, sc_a_ba_pa_nc, cs_a_ab_iq, cs_a_ab_iq_nc, &
      sc_b_ab_pa, sc_b_ab_pa_nc, cs_b_ba_iq, cs_b_ba_iq_nc, &
      cs_ea_a_q, cs_ea_a_q_nc, sc_ip_b_p, sc_ip_b_p_nc, sc_ip_a_p, sc_ip_a_p_nc, cs_ea_b_q, cs_ea_b_q_nc, &
      aca_a_right, aca_a_right_nc, aca_b_right, aca_b_right_nc, aca_a_left, aca_a_left_nc, aca_b_left, aca_b_left_nc, &
      scs_a_bb_iq, scs_a_bb_iq_nc, scs_a_bb_pa, scs_a_bb_pa_nc, scs_b_aa_pa, scs_b_aa_pa_nc, &
      ca_ea_a_right, ca_ea_a_right_nc, ac_ip_b_right, ac_ip_b_right_nc, ac_ip_a_right, ac_ip_a_right_nc, &
      ca_ea_b_right, ca_ea_b_right_nc, ca_ea_a_left, ca_ea_a_left_nc, ac_ip_b_left, ac_ip_b_left_nc, &
      ac_ip_a_left, ac_ip_a_left_nc, ca_ea_b_left, ca_ea_b_left_nc, &
      out, err_flag)

    !f2py intent(in) :: pos2, pos3, pos4
    !f2py intent(in) :: a, b, c, d, v
    !f2py intent(in) :: nbas_a
    !f2py intent(in) :: sc_a_ba, cs_a_ab, sc_b_ab, cs_b_ba, cs_ea_a, sc_ip_a, sc_ip_b, cs_ea_b
    !f2py intent(in) :: sc_a_ba_pa, cs_a_ab_iq, sc_b_ab_pa, cs_b_ba_iq
    !f2py intent(in) :: cs_ea_a_q, sc_ip_b_p, sc_ip_a_p, cs_ea_b_q
    !f2py intent(in) :: aca_a_right, aca_b_right, aca_a_left, aca_b_left
    !f2py intent(in) :: scs_a_bb_iq, scs_a_bb_pa, scs_b_aa_pa
    !f2py intent(in) :: ca_ea_a_right, ac_ip_b_right, ac_ip_a_right, ca_ea_b_right, ca_ea_a_left, ac_ip_b_left
    !f2py intent(in) :: ac_ip_a_left, ca_ea_b_left
    !f2py intent(in) :: sc_a_ba_nc, cs_a_ab_nc, sc_b_ab_nc, cs_b_ba_nc
    !f2py intent(in) :: cs_ea_a_nc, sc_ip_a_nc, sc_ip_b_nc, cs_ea_b_nc
    !f2py intent(in) :: sc_a_ba_pa_nc, cs_a_ab_iq_nc, sc_b_ab_pa_nc, cs_b_ba_iq_nc
    !f2py intent(in) :: cs_ea_a_q_nc, sc_ip_b_p_nc, sc_ip_a_p_nc, cs_ea_b_q_nc
    !f2py intent(in) :: aca_a_right_nc, aca_b_right_nc, aca_a_left_nc, aca_b_left_nc
    !f2py intent(in) :: scs_a_bb_iq_nc, scs_a_bb_pa_nc, scs_b_aa_pa_nc
    !f2py intent(in) :: ca_ea_a_right_nc, ac_ip_b_right_nc, ac_ip_a_right_nc, ca_ea_b_right_nc
    !f2py intent(in) :: ca_ea_a_left_nc, ac_ip_b_left_nc, ac_ip_a_left_nc, ca_ea_b_left_nc
    !f2py intent(out) :: out, err_flag

    integer*8, intent(in) :: pos2(:), pos3(:), pos4(:)
    integer*2, intent(in) :: a(:), b(:), c(:), d(:)
    double precision, intent(in) :: v(:)
    integer, intent(in) :: nbas_a

    double precision, intent(in) :: sc_a_ba(:), cs_a_ab(:), sc_b_ab(:), cs_b_ba(:)
    double precision, intent(in) :: cs_ea_a(:), sc_ip_a(:), sc_ip_b(:), cs_ea_b(:)
    double precision, intent(in) :: sc_a_ba_pa(:), cs_a_ab_iq(:), sc_b_ab_pa(:), cs_b_ba_iq(:)
    double precision, intent(in) :: cs_ea_a_q(:), sc_ip_b_p(:), sc_ip_a_p(:), cs_ea_b_q(:)
    double precision, intent(in) :: aca_a_right(:), aca_b_right(:), aca_a_left(:), aca_b_left(:)
    double precision, intent(in) :: scs_a_bb_iq(:), scs_a_bb_pa(:), scs_b_aa_pa(:)
    double precision, intent(in) :: ca_ea_a_right(:), ac_ip_b_right(:), ac_ip_a_right(:)
    double precision, intent(in) :: ca_ea_b_right(:), ca_ea_a_left(:), ac_ip_b_left(:), ac_ip_a_left(:), ca_ea_b_left(:)

    integer, intent(in) :: sc_a_ba_nc, cs_a_ab_nc, sc_b_ab_nc, cs_b_ba_nc
    integer, intent(in) :: cs_ea_a_nc, sc_ip_a_nc, sc_ip_b_nc, cs_ea_b_nc
    integer, intent(in) :: sc_a_ba_pa_nc, cs_a_ab_iq_nc, sc_b_ab_pa_nc, cs_b_ba_iq_nc
    integer, intent(in) :: cs_ea_a_q_nc, sc_ip_b_p_nc, sc_ip_a_p_nc, cs_ea_b_q_nc
    integer, intent(in) :: aca_a_right_nc, aca_b_right_nc, aca_a_left_nc, aca_b_left_nc
    integer, intent(in) :: scs_a_bb_iq_nc, scs_a_bb_pa_nc, scs_b_aa_pa_nc
    integer, intent(in) :: ca_ea_a_right_nc, ac_ip_b_right_nc, ac_ip_a_right_nc, ca_ea_b_right_nc
    integer, intent(in) :: ca_ea_a_left_nc, ac_ip_b_left_nc, ac_ip_a_left_nc, ca_ea_b_left_nc

    double precision, intent(out) :: out(82)
    integer, intent(out) :: err_flag

    double precision :: f21_c, f21_x, f43_c, f43_x, f12_c, f12_x, f34_c, f34_x
    double precision :: f21_c_2s, f21_x_2s, f12_c_2s, f12_x_2s
    double precision :: f31_c_loc_occ, f31_x_loc_occ, f13_c_loc_virt, f13_x_loc_virt, f31_c_ea, f31_x_ea
    double precision :: f41_c_loc_occ, f41_x_loc_occ, f14_c_loc_virt, f14_x_loc_virt, f14_c_ip, f14_x_ip
    double precision :: f32_c_loc_occ, f32_x_loc_occ
    double precision :: f42_c_loc_occ, f42_x_loc_occ, f42_c_ea, f42_x_ea
    double precision :: f23_c_loc_virt, f23_x_loc_virt, f23_c_ip, f23_x_ip
    double precision :: f24_c_loc_virt, f24_x_loc_virt

    integer*8 :: t
    integer*8 :: p
    integer :: i, j, k, l
    integer :: a0, a1, b0, b1
    logical :: ok
    double precision :: value
    double precision, parameter :: cpref = 4.0d0
    double precision, parameter :: xpref = -2.0d0

    err_flag = 0
    out = 0.0d0

    f21_c = 0.0d0; f21_x = 0.0d0; f43_c = 0.0d0; f43_x = 0.0d0
    f12_c = 0.0d0; f12_x = 0.0d0; f34_c = 0.0d0; f34_x = 0.0d0
    f21_c_2s = 0.0d0; f21_x_2s = 0.0d0; f12_c_2s = 0.0d0; f12_x_2s = 0.0d0
    f31_c_loc_occ = 0.0d0; f31_x_loc_occ = 0.0d0; f13_c_loc_virt = 0.0d0; f13_x_loc_virt = 0.0d0
    f31_c_ea = 0.0d0; f31_x_ea = 0.0d0
    f41_c_loc_occ = 0.0d0; f41_x_loc_occ = 0.0d0; f14_c_loc_virt = 0.0d0; f14_x_loc_virt = 0.0d0
    f14_c_ip = 0.0d0; f14_x_ip = 0.0d0
    f32_c_loc_occ = 0.0d0; f32_x_loc_occ = 0.0d0
    f42_c_loc_occ = 0.0d0; f42_x_loc_occ = 0.0d0; f42_c_ea = 0.0d0; f42_x_ea = 0.0d0
    f23_c_loc_virt = 0.0d0; f23_x_loc_virt = 0.0d0; f23_c_ip = 0.0d0; f23_x_ip = 0.0d0
    f24_c_loc_virt = 0.0d0; f24_x_loc_virt = 0.0d0

    ! -------------------------
    ! unique_count == 4 block
    ! -------------------------
    do t = 1_8, int(size(pos4), 8)
      p = pos4(t) + 1_8
      i = u16_from_i16(a(p))
      j = u16_from_i16(b(p))
      k = u16_from_i16(c(p))
      l = u16_from_i16(d(p))
      value = v(p)
      if (j > nbas_a) then
        call split_ab(i, j, k, l, nbas_a, a0, a1, b0, b1, ok)
        if (.not. ok) then
          err_flag = 1
          cycle
        end if
        call accum_u4_coul(a0, a1, b0, b1, value)
        call accum_new_u4_coul(a0, a1, b0, b1, value)
      else
        if (i < k) then
          call split_ab(k, l, i, j, nbas_a, a0, a1, b0, b1, ok)
        else
          call split_ab(i, j, k, l, nbas_a, a0, a1, b0, b1, ok)
        end if
        if (.not. ok) then
          err_flag = 1
          cycle
        end if
        call accum_u4_exch(a0, a1, b0, b1, value)
        call accum_new_u4_exch(a0, a1, b0, b1, value)
      end if
    end do

    ! -------------------------
    ! unique_count == 3 block
    ! -------------------------
    do t = 1_8, int(size(pos3), 8)
      p = pos3(t) + 1_8
      i = u16_from_i16(a(p))
      j = u16_from_i16(b(p))
      k = u16_from_i16(c(p))
      l = u16_from_i16(d(p))
      value = v(p)
      call split_ab(i, j, k, l, nbas_a, a0, a1, b0, b1, ok)
      if (.not. ok) then
        err_flag = 1
        cycle
      end if

      if (j > nbas_a) then
        if (i == j) then
          call accum_u3_coul_with_ov_a(a0, a1, b0, b1, value)
          call accum_u3_coul(which_frag_is_a=.true., a0=a0, a1=a1, b0=b0, b1=b1, value=value)
          call accum_new_u3_coul(.true., a0, a1, b0, b1, value)
        else
          call accum_u3_coul_with_ov_b(a0, a1, b0, b1, value)
          call accum_u3_coul(which_frag_is_a=.false., a0=a0, a1=a1, b0=b0, b1=b1, value=value)
          call accum_new_u3_coul(.false., a0, a1, b0, b1, value)
        end if
      else
        if (i == k) then
          call accum_u3_exch_with_ov_a(a0, a1, b0, b1, value)
          call accum_u3_exch(which_frag_is_a=.true., a0=a0, a1=a1, b0=b0, b1=b1, value=value)
          call accum_new_u3_exch(.true., a0, a1, b0, b1, value)
        else
          call accum_u3_exch_with_ov_b(a0, a1, b0, b1, value)
          call accum_u3_exch(which_frag_is_a=.false., a0=a0, a1=a1, b0=b0, b1=b1, value=value)
          call accum_new_u3_exch(.false., a0, a1, b0, b1, value)
        end if
      end if
    end do

    ! -------------------------
    ! unique_count == 2 block
    ! -------------------------
    do t = 1_8, int(size(pos2), 8)
      p = pos2(t) + 1_8
      i = u16_from_i16(a(p))
      j = u16_from_i16(b(p))
      k = u16_from_i16(c(p))
      l = u16_from_i16(d(p))
      value = v(p)
      call split_ab(i, j, k, l, nbas_a, a0, a1, b0, b1, ok)
      if (.not. ok) then
        err_flag = 1
        cycle
      end if
      if (i == j) then
        call accum_u2_coul(a0, a1, b0, b1, value)
        call accum_new_u2_coul(a0, a1, b0, b1, value)
      else
        call accum_u2_exch(a0, a1, b0, b1, value)
        call accum_new_u2_exch(a0, a1, b0, b1, value)
      end if
    end do

    out(1)  = f21_c
    out(2)  = f21_x
    out(3)  = f43_c
    out(4)  = f43_x
    out(5)  = f12_c
    out(6)  = f12_x
    out(7)  = f34_c
    out(8)  = f34_x
    out(9)  = f31_c_loc_occ
    out(10) = f31_x_loc_occ
    out(11) = f13_c_loc_virt
    out(12) = f13_x_loc_virt
    out(13) = f31_c_ea
    out(14) = f31_x_ea
    out(15) = f41_c_loc_occ
    out(16) = f41_x_loc_occ
    out(17) = f14_c_loc_virt
    out(18) = f14_x_loc_virt
    out(19) = f14_c_ip
    out(20) = f14_x_ip
    out(21) = f21_c_2s
    out(22) = f21_x_2s
    out(23) = f12_c_2s
    out(24) = f12_x_2s
    out(25) = f32_c_loc_occ
    out(26) = f32_x_loc_occ
    out(27) = f42_c_loc_occ
    out(28) = f42_x_loc_occ
    out(29) = f42_c_ea
    out(30) = f42_x_ea
    out(31) = f23_c_loc_virt
    out(32) = f23_x_loc_virt
    out(33) = f23_c_ip
    out(34) = f23_x_ip
    out(35) = f24_c_loc_virt
    out(36) = f24_x_loc_virt

  contains

    subroutine accum_u2_coul(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      f31_x_loc_occ = f31_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, b0, a0) * &
          get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val
      f31_c_ea = f31_c_ea + cpref * get2(aca_a_right, aca_a_right_nc, a0, a1) * &
          get2(cs_ea_a, cs_ea_a_nc, 0, b0) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val

      f21_x_2s = f21_x_2s + xpref * get2(sc_a_ba, sc_a_ba_nc, b0, a1) * &
          get2(cs_b_ba, cs_b_ba_nc, b1, a0) * val
      f12_x_2s = f12_x_2s + xpref * get2(sc_b_ab, sc_b_ab_nc, a0, b1) * &
          get2(cs_a_ab, cs_a_ab_nc, a1, b0) * val

       f42_x_loc_occ = f42_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, a0, b0) * &
           get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * val
      f42_c_ea = f42_c_ea + cpref * get2(aca_b_right, aca_b_right_nc, b0, b1) * &
          get2(cs_ea_b, cs_ea_b_nc, 0, a0) * get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * val

      f14_x_loc_virt = f14_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, a0, b0) * &
          get2(ac_ip_a_right, ac_ip_a_right_nc, a1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val
      f14_c_ip = f14_c_ip + cpref * get2(aca_a_left, aca_a_left_nc, a0, a1) * &
          get2(sc_ip_a, sc_ip_a_nc, b0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val

      f23_x_loc_virt = f23_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, b0, a0) * &
          get2(ac_ip_b_right, ac_ip_b_right_nc, b1, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val
      f23_c_ip = f23_c_ip + cpref * get2(aca_b_left, aca_b_left_nc, b0, b1) * &
          get2(sc_ip_b, sc_ip_b_nc, a0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val

      f21_c = f21_c + cpref * get2(aca_a_right, aca_a_right_nc, a0, a1) * &
          get2(aca_b_left, aca_b_left_nc, b0, b1) * val
      f12_c = f12_c + cpref * get2(aca_a_left, aca_a_left_nc, a0, a1) * &
          get2(aca_b_right, aca_b_right_nc, b0, b1) * val
    end subroutine accum_u2_coul

    subroutine accum_u2_exch(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val

      f21_c_2s = f21_c_2s + cpref * get2(sc_a_ba, sc_a_ba_nc, b0, a1) * &
          get2(cs_b_ba, cs_b_ba_nc, b1, a0) * val
      f12_c_2s = f12_c_2s + cpref * get2(sc_b_ab, sc_b_ab_nc, a0, b1) * &
          get2(cs_a_ab, cs_a_ab_nc, a1, b0) * val

      f31_c_loc_occ = f31_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, b0, a0) * &
          get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val
      f31_x_ea = f31_x_ea + xpref * get2(aca_a_right, aca_a_right_nc, a0, a1) * &
          get2(cs_ea_a, cs_ea_a_nc, 0, b0) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val

      f13_c_loc_virt = f13_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, a0, b0) * &
          get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * get2(ac_ip_b_right, ac_ip_b_right_nc, b1, 0) * val
      f13_x_loc_virt = f13_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, a0, b0) * &
          get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * get2(ac_ip_b_right, ac_ip_b_right_nc, b1, 0) * val

      f24_c_loc_virt = f24_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, b0, a0) * &
          get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * get2(ac_ip_a_right, ac_ip_a_right_nc, a1, 0) * val
      f24_x_loc_virt = f24_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, b0, a0) * &
          get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * get2(ac_ip_a_right, ac_ip_a_right_nc, a1, 0) * val

      f42_c_loc_occ = f42_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, a0, b0) * &
          get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * val
      f42_x_ea = f42_x_ea + xpref * get2(aca_b_right, aca_b_right_nc, b0, b1) * &
          get2(cs_ea_b, cs_ea_b_nc, 0, a0) * get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * val

      f14_c_loc_virt = f14_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, a0, b0) * &
          get2(ac_ip_a_right, ac_ip_a_right_nc, a1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val
      f14_x_ip = f14_x_ip + xpref * get2(aca_a_left, aca_a_left_nc, a0, a1) * &
          get2(sc_ip_a, sc_ip_a_nc, b0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val

      f41_c_loc_occ = f41_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, b0, a0) * &
          get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * val
      f41_x_loc_occ = f41_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, b0, a0) * &
          get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * val

      f23_c_loc_virt = f23_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, b0, a0) * &
          get2(ac_ip_b_right, ac_ip_b_right_nc, b1, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val
      f23_x_ip = f23_x_ip + xpref * get2(aca_b_left, aca_b_left_nc, b0, b1) * &
          get2(sc_ip_b, sc_ip_b_nc, a0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val

      f32_c_loc_occ = f32_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, a0, b0) * &
          get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * val
      f32_x_loc_occ = f32_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, a0, b0) * &
          get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * val

      f21_x = f21_x + xpref * get2(aca_a_right, aca_a_right_nc, a0, a1) * &
          get2(aca_b_left, aca_b_left_nc, b0, b1) * val
      f12_x = f12_x + xpref * get2(aca_a_left, aca_a_left_nc, a0, a1) * &
          get2(aca_b_right, aca_b_right_nc, b0, b1) * val

      f43_x = f43_x + xpref * get2(ac_ip_a_left, ac_ip_a_left_nc, a0, 0) * &
          get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * &
          get2(ac_ip_b_right, ac_ip_b_right_nc, b0, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * val
      f43_c = f43_c + cpref * get2(ac_ip_a_left, ac_ip_a_left_nc, a0, 0) * &
          get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * &
          get2(ac_ip_b_right, ac_ip_b_right_nc, b0, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * val

      f34_x = f34_x + xpref * get2(ac_ip_a_right, ac_ip_a_right_nc, a0, 0) * &
          get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * &
          get2(ac_ip_b_left, ac_ip_b_left_nc, b0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val
      f34_c = f34_c + cpref * get2(ac_ip_a_right, ac_ip_a_right_nc, a0, 0) * &
          get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * &
          get2(ac_ip_b_left, ac_ip_b_left_nc, b0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val
    end subroutine accum_u2_exch

    subroutine accum_u4_coul(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: pa0, pa1, pb0, pb1
      integer :: ia, ia2

      do ia = 1, 2
        if (ia == 1) then
          pa0 = a1; pa1 = a0
        else
          pa0 = a0; pa1 = a1
        end if
        do ia2 = 1, 2
          if (ia2 == 1) then
            pb0 = b1; pb1 = b0
          else
            pb0 = b0; pb1 = b1
          end if

          f21_c = f21_c + cpref * get2(aca_a_right, aca_a_right_nc, pa0, pa1) * &
              get2(aca_b_left, aca_b_left_nc, pb0, pb1) * val
          f12_c = f12_c + cpref * get2(aca_a_left, aca_a_left_nc, pa0, pa1) * &
              get2(aca_b_right, aca_b_right_nc, pb0, pb1) * val

          f31_c_ea = f31_c_ea + cpref * get2(aca_a_right, aca_a_right_nc, pa0, pa1) * &
              get2(cs_ea_a, cs_ea_a_nc, 0, pb1) * get2(ac_ip_b_left, ac_ip_b_left_nc, pb0, 0) * val
          f31_x_loc_occ = f31_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, pb0, pa0) * &
              get2(ca_ea_a_left, ca_ea_a_left_nc, 0, pa1) * get2(ac_ip_b_left, ac_ip_b_left_nc, pb1, 0) * val

          f42_c_ea = f42_c_ea + cpref * get2(aca_b_right, aca_b_right_nc, pb1, pb0) * &
              get2(cs_ea_b, cs_ea_b_nc, 0, pa1) * get2(ac_ip_a_left, ac_ip_a_left_nc, pa0, 0) * val
          f42_x_loc_occ = f42_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, pa1, pb1) * &
              get2(ca_ea_b_left, ca_ea_b_left_nc, 0, pb0) * get2(ac_ip_a_left, ac_ip_a_left_nc, pa0, 0) * val

          f21_x_2s = f21_x_2s + xpref * get2(sc_a_ba, sc_a_ba_nc, pb0, pa0) * &
              get2(cs_b_ba, cs_b_ba_nc, pb1, pa1) * val
          f12_x_2s = f12_x_2s + xpref * get2(sc_b_ab, sc_b_ab_nc, pa0, pb0) * &
              get2(cs_a_ab, cs_a_ab_nc, pa1, pb1) * val

          f14_x_loc_virt = f14_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, pa0, pb0) * &
              get2(ac_ip_a_right, ac_ip_a_right_nc, pa1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb1) * val
          f14_c_ip = f14_c_ip + cpref * get2(aca_a_left, aca_a_left_nc, pa0, pa1) * &
              get2(sc_ip_a, sc_ip_a_nc, pb0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb1) * val

          f23_x_loc_virt = f23_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, pb0, pa0) * &
              get2(ac_ip_b_right, ac_ip_b_right_nc, pb1, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa1) * val
          f23_c_ip = f23_c_ip + cpref * get2(aca_b_left, aca_b_left_nc, pb0, pb1) * &
              get2(sc_ip_b, sc_ip_b_nc, pa0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa1) * val
        end do
      end do
    end subroutine accum_u4_coul

    subroutine accum_u4_exch(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: pa0, pa1, pb0, pb1
      integer :: idx

      do idx = 1, 2
        if (idx == 1) then
          pa0 = a1; pa1 = a0
          pb0 = b1; pb1 = b0
        else
          pa0 = a0; pa1 = a1
          pb0 = b0; pb1 = b1
        end if

        f13_c_loc_virt = f13_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, pa0, pb0) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa1) * get2(ac_ip_b_right, ac_ip_b_right_nc, pb1, 0) * val
        f13_x_loc_virt = f13_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, pa0, pb1) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa1) * get2(ac_ip_b_right, ac_ip_b_right_nc, pb0, 0) * val

        f31_c_loc_occ = f31_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, pb0, pa0) * &
            get2(ca_ea_a_left, ca_ea_a_left_nc, 0, pa1) * get2(ac_ip_b_left, ac_ip_b_left_nc, pb1, 0) * val
        f31_x_ea = f31_x_ea + xpref * get2(aca_a_right, aca_a_right_nc, pa0, pa1) * &
            get2(cs_ea_a, cs_ea_a_nc, 0, pb1) * get2(ac_ip_b_left, ac_ip_b_left_nc, pb0, 0) * val

        f21_c_2s = f21_c_2s + cpref * get2(sc_a_ba, sc_a_ba_nc, pb0, pa0) * &
            get2(cs_b_ba, cs_b_ba_nc, pb1, pa1) * val
        f12_c_2s = f12_c_2s + cpref * get2(sc_b_ab, sc_b_ab_nc, pa0, pb0) * &
            get2(cs_a_ab, cs_a_ab_nc, pa1, pb1) * val

        f42_c_loc_occ = f42_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, pa1, pb1) * &
            get2(ca_ea_b_left, ca_ea_b_left_nc, 0, pb0) * get2(ac_ip_a_left, ac_ip_a_left_nc, pa0, 0) * val
        f42_x_ea = f42_x_ea + xpref * get2(aca_b_right, aca_b_right_nc, pb0, pb1) * &
            get2(cs_ea_b, cs_ea_b_nc, 0, pa1) * get2(ac_ip_a_left, ac_ip_a_left_nc, pa0, 0) * val

        f24_c_loc_virt = f24_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, pb1, pa1) * &
            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb0) * get2(ac_ip_a_right, ac_ip_a_right_nc, pa0, 0) * val
        f24_x_loc_virt = f24_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, pb1, pa0) * &
            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb0) * get2(ac_ip_a_right, ac_ip_a_right_nc, pa1, 0) * val

        f14_c_loc_virt = f14_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, pa0, pb0) * &
            get2(ac_ip_a_right, ac_ip_a_right_nc, pa1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb1) * val
        f14_x_ip = f14_x_ip + xpref * get2(aca_a_left, aca_a_left_nc, pa0, pa1) * &
            get2(sc_ip_a, sc_ip_a_nc, pb0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb1) * val

        f41_c_loc_occ = f41_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, pb0, pa0) * &
            get2(ac_ip_a_left, ac_ip_a_left_nc, pa1, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, pb1) * val
        f41_x_loc_occ = f41_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, pb0, pa1) * &
            get2(ac_ip_a_left, ac_ip_a_left_nc, pa0, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, pb1) * val

        f32_c_loc_occ = f32_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, pa0, pb0) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, pb1, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, pa1) * val
        f32_x_loc_occ = f32_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, pa0, pb1) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, pb0, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, pa1) * val

        f23_c_loc_virt = f23_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, pb1, pa1) * &
            get2(ac_ip_b_right, ac_ip_b_right_nc, pb0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa0) * val
        f23_x_ip = f23_x_ip + xpref * get2(aca_b_left, aca_b_left_nc, pb0, pb1) * &
            get2(sc_ip_b, sc_ip_b_nc, pa0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa1) * val

        f43_x = f43_x + xpref * get2(ac_ip_a_left, ac_ip_a_left_nc, pa0, 0) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa1) * &
            get2(ac_ip_b_right, ac_ip_b_right_nc, pb0, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, pb1) * val
        f43_c = f43_c + cpref * get2(ac_ip_a_left, ac_ip_a_left_nc, pa1, 0) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, pa0) * &
            get2(ac_ip_b_right, ac_ip_b_right_nc, pb0, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, pb1) * val

        f34_x = f34_x + xpref * get2(ac_ip_a_right, ac_ip_a_right_nc, pa0, 0) * &
            get2(ca_ea_a_left, ca_ea_a_left_nc, 0, pa1) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, pb0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb1) * val
        
        f34_c = f34_c + cpref * get2(ac_ip_a_right, ac_ip_a_right_nc, pa1, 0) * &
            get2(ca_ea_a_left, ca_ea_a_left_nc, 0, pa0) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, pb0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, pb1) * val
            
        f21_x = f21_x + xpref * get2(aca_a_right, aca_a_right_nc, pa0, pa1) * &
            get2(aca_b_left, aca_b_left_nc, pb0, pb1) * val
        f12_x = f12_x + xpref * get2(aca_a_left, aca_a_left_nc, pa0, pa1) * &
            get2(aca_b_right, aca_b_right_nc, pb0, pb1) * val
      end do
    end subroutine accum_u4_exch

    subroutine accum_u3_coul_with_ov_a(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: e0, e1
      integer :: idx

      do idx = 1, 2
        if (idx == 1) then
          e0 = a1; e1 = a0
        else
          e0 = a0; e1 = a1
        end if

        f31_c_ea = f31_c_ea + cpref * get2(aca_a_right, aca_a_right_nc, e0, e1) * &
            get2(cs_ea_a, cs_ea_a_nc, 0, b0) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val
        f31_x_loc_occ = f31_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, b0, e0) * &
            get2(ca_ea_a_left, ca_ea_a_left_nc, 0, e1) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val

        f21_x_2s = f21_x_2s + xpref * get2(sc_a_ba, sc_a_ba_nc, b0, e0) * &
            get2(cs_b_ba, cs_b_ba_nc, b1, e1) * val
        f12_x_2s = f12_x_2s + xpref * get2(sc_b_ab, sc_b_ab_nc, e0, b0) * &
            get2(cs_a_ab, cs_a_ab_nc, e1, b1) * val

        f42_c_ea = f42_c_ea + cpref * get2(aca_b_right, aca_b_right_nc, b1, b0) * &
            get2(cs_ea_b, cs_ea_b_nc, 0, e1) * get2(ac_ip_a_left, ac_ip_a_left_nc, e0, 0) * val
        f42_x_loc_occ = f42_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, e0, b0) * &
            get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * get2(ac_ip_a_left, ac_ip_a_left_nc, e1, 0) * val

        f14_c_ip = f14_c_ip + cpref * get2(aca_a_left, aca_a_left_nc, e0, e1) * &
            get2(sc_ip_a, sc_ip_a_nc, b0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val
        f14_x_loc_virt = f14_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, e0, b0) * &
            get2(ac_ip_a_right, ac_ip_a_right_nc, e1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val

        f23_c_ip = f23_c_ip + cpref * get2(aca_b_left, aca_b_left_nc, b0, b1) * &
            get2(sc_ip_b, sc_ip_b_nc, e0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e1) * val
        f23_x_loc_virt = f23_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, b1, e1) * &
            get2(ac_ip_b_right, ac_ip_b_right_nc, b0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e0) * val
      end do
    end subroutine accum_u3_coul_with_ov_a

    subroutine accum_u3_coul_with_ov_b(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: e0, e1
      integer :: idx

      do idx = 1, 2
        if (idx == 1) then
          e0 = b1; e1 = b0
        else
          e0 = b0; e1 = b1
        end if

        f31_c_ea = f31_c_ea + cpref * get2(aca_a_right, aca_a_right_nc, a0, a1) * &
            get2(cs_ea_a, cs_ea_a_nc, 0, e0) * get2(ac_ip_b_left, ac_ip_b_left_nc, e1, 0) * val
        f31_x_loc_occ = f31_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, e0, a0) * &
            get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * get2(ac_ip_b_left, ac_ip_b_left_nc, e1, 0) * val

        f21_x_2s = f21_x_2s + xpref * get2(sc_a_ba, sc_a_ba_nc, e0, a0) * &
            get2(cs_b_ba, cs_b_ba_nc, e1, a1) * val
        f12_x_2s = f12_x_2s + xpref * get2(sc_b_ab, sc_b_ab_nc, a0, e0) * &
            get2(cs_a_ab, cs_a_ab_nc, a1, e1) * val

        f42_c_ea = f42_c_ea + cpref * get2(aca_b_right, aca_b_right_nc, e1, e0) * &
            get2(cs_ea_b, cs_ea_b_nc, 0, a1) * get2(ac_ip_a_left, ac_ip_a_left_nc, a0, 0) * val
        f42_x_loc_occ = f42_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, a0, e0) * &
            get2(ca_ea_b_left, ca_ea_b_left_nc, 0, e1) * get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * val

        f14_c_ip = f14_c_ip + cpref * get2(aca_a_left, aca_a_left_nc, a0, a1) * &
            get2(sc_ip_a, sc_ip_a_nc, e0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e1) * val
        f14_x_loc_virt = f14_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, a0, e0) * &
            get2(ac_ip_a_right, ac_ip_a_right_nc, a1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e1) * val

        f23_c_ip = f23_c_ip + cpref * get2(aca_b_left, aca_b_left_nc, e0, e1) * &
            get2(sc_ip_b, sc_ip_b_nc, a0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val
        f23_x_loc_virt = f23_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, e0, a0) * &
            get2(ac_ip_b_right, ac_ip_b_right_nc, e1, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val
      end do
    end subroutine accum_u3_coul_with_ov_b

    subroutine accum_u3_coul(which_frag_is_a, a0, a1, b0, b1, value)
      logical, intent(in) :: which_frag_is_a
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: value
      integer :: e0, e1
      integer :: idx
      double precision :: term21, term12

      if (which_frag_is_a) then
        term21 = value * get2(aca_b_left, aca_b_left_nc, b0, b1)
        term12 = value * get2(aca_b_right, aca_b_right_nc, b0, b1)
        do idx = 1, 2
          if (idx == 1) then
            e0 = a1; e1 = a0
          else
            e0 = a0; e1 = a1
          end if
          f21_c = f21_c + cpref * get2(aca_a_right, aca_a_right_nc, e0, e1) * term21
          f12_c = f12_c + cpref * get2(aca_a_left, aca_a_left_nc, e0, e1) * term12
        end do
      else
        term21 = value * get2(aca_a_right, aca_a_right_nc, a0, a1)
        term12 = value * get2(aca_a_left, aca_a_left_nc, a0, a1)
        do idx = 1, 2
          if (idx == 1) then
            e0 = b1; e1 = b0
          else
            e0 = b0; e1 = b1
          end if
          f21_c = f21_c + cpref * get2(aca_b_left, aca_b_left_nc, e0, e1) * term21
          f12_c = f12_c + cpref * get2(aca_b_right, aca_b_right_nc, e0, e1) * term12
        end do
      end if
    end subroutine accum_u3_coul

    subroutine accum_u3_exch_with_ov_a(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: e0, e1
      integer :: idx

      do idx = 1, 2
        if (idx == 1) then
          e0 = a1; e1 = a0
        else
          e0 = a0; e1 = a1
        end if

        f31_c_loc_occ = f31_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, b0, e0) * &
            get2(ca_ea_a_left, ca_ea_a_left_nc, 0, e1) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val
        f31_x_ea = f31_x_ea + xpref * get2(aca_a_right, aca_a_right_nc, e0, e1) * &
            get2(cs_ea_a, cs_ea_a_nc, 0, b0) * get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * val

        f13_c_loc_virt = f13_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, e0, b0) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e1) * get2(ac_ip_b_right, ac_ip_b_right_nc, b1, 0) * val
        f13_x_loc_virt = f13_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, e0, b0) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e1) * get2(ac_ip_b_right, ac_ip_b_right_nc, b1, 0) * val

        f42_c_loc_occ = f42_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, e0, b0) * &
            get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * get2(ac_ip_a_left, ac_ip_a_left_nc, e1, 0) * val
        f42_x_ea = f42_x_ea + xpref * get2(aca_b_right, aca_b_right_nc, b1, b0) * &
            get2(cs_ea_b, cs_ea_b_nc, 0, e1) * get2(ac_ip_a_left, ac_ip_a_left_nc, e0, 0) * val

        f24_c_loc_virt = f24_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, b1, e1) * &
            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b0) * get2(ac_ip_a_right, ac_ip_a_right_nc, e0, 0) * val
        f24_x_loc_virt = f24_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, b1, e1) * &
            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b0) * get2(ac_ip_a_right, ac_ip_a_right_nc, e0, 0) * val

        f21_c_2s = f21_c_2s + cpref * get2(sc_a_ba, sc_a_ba_nc, b0, e0) * &
            get2(cs_b_ba, cs_b_ba_nc, b1, e1) * val
        f12_c_2s = f12_c_2s + cpref * get2(sc_b_ab, sc_b_ab_nc, e0, b0) * &
            get2(cs_a_ab, cs_a_ab_nc, e1, b1) * val

        f14_c_loc_virt = f14_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, e0, b0) * &
            get2(ac_ip_a_right, ac_ip_a_right_nc, e1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val
        f14_x_ip = f14_x_ip + xpref * get2(aca_a_left, aca_a_left_nc, e0, e1) * &
            get2(sc_ip_a, sc_ip_a_nc, b0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b1) * val

        f41_c_loc_occ = f41_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, b0, e0) * &
            get2(ac_ip_a_left, ac_ip_a_left_nc, e1, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * val
        f41_x_loc_occ = f41_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, b0, e0) * &
            get2(ac_ip_a_left, ac_ip_a_left_nc, e1, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b1) * val

        f32_c_loc_occ = f32_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, e0, b0) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, e1) * val
        f32_x_loc_occ = f32_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, e0, b0) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, b1, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, e1) * val

        f23_c_loc_virt = f23_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, b1, e1) * &
            get2(ac_ip_b_right, ac_ip_b_right_nc, b0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e0) * val
        f23_x_ip = f23_x_ip + xpref * get2(aca_b_left, aca_b_left_nc, b0, b1) * &
            get2(sc_ip_b, sc_ip_b_nc, e0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e1) * val
      end do
    end subroutine accum_u3_exch_with_ov_a

    subroutine accum_u3_exch_with_ov_b(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: e0, e1
      integer :: idx

      do idx = 1, 2
        if (idx == 1) then
          e0 = b1; e1 = b0
        else
          e0 = b0; e1 = b1
        end if

        f13_c_loc_virt = f13_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, a0, e0) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * get2(ac_ip_b_right, ac_ip_b_right_nc, e1, 0) * val
        f13_x_loc_virt = f13_x_loc_virt + xpref * get2(cs_a_ab, cs_a_ab_nc, a0, e0) * &
            get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * get2(ac_ip_b_right, ac_ip_b_right_nc, e1, 0) * val

        f31_c_loc_occ = f31_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, e0, a0) * &
            get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * get2(ac_ip_b_left, ac_ip_b_left_nc, e1, 0) * val
        f31_x_ea = f31_x_ea + xpref * get2(aca_a_right, aca_a_right_nc, a0, a1) * &
            get2(cs_ea_a, cs_ea_a_nc, 0, e0) * get2(ac_ip_b_left, ac_ip_b_left_nc, e1, 0) * val

        f42_c_loc_occ = f42_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, a0, e0) * &
            get2(ca_ea_b_left, ca_ea_b_left_nc, 0, e1) * get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * val
        f42_x_ea = f42_x_ea + xpref * get2(aca_b_right, aca_b_right_nc, e0, e1) * &
            get2(cs_ea_b, cs_ea_b_nc, 0, a0) * get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * val

        f24_c_loc_virt = f24_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, e1, a1) * &
            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e0) * get2(ac_ip_a_right, ac_ip_a_right_nc, a0, 0) * val
        f24_x_loc_virt = f24_x_loc_virt + xpref * get2(cs_b_ba, cs_b_ba_nc, e1, a1) * &
            get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e0) * get2(ac_ip_a_right, ac_ip_a_right_nc, a0, 0) * val

        f14_c_loc_virt = f14_c_loc_virt + cpref * get2(cs_a_ab, cs_a_ab_nc, a0, e0) * &
            get2(ac_ip_a_right, ac_ip_a_right_nc, a1, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e1) * val
        f14_x_ip = f14_x_ip + xpref * get2(aca_a_left, aca_a_left_nc, a0, a1) * &
            get2(sc_ip_a, sc_ip_a_nc, e0, 0) * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e1) * val

        f41_c_loc_occ = f41_c_loc_occ + cpref * get2(sc_a_ba, sc_a_ba_nc, e0, a0) * &
            get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, e1) * val
        f41_x_loc_occ = f41_x_loc_occ + xpref * get2(sc_a_ba, sc_a_ba_nc, e0, a0) * &
            get2(ac_ip_a_left, ac_ip_a_left_nc, a1, 0) * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, e1) * val

        f21_c_2s = f21_c_2s + cpref * get2(sc_a_ba, sc_a_ba_nc, e0, a0) * &
            get2(cs_b_ba, cs_b_ba_nc, e1, a1) * val
        f12_c_2s = f12_c_2s + cpref * get2(sc_b_ab, sc_b_ab_nc, a0, e0) * &
            get2(cs_a_ab, cs_a_ab_nc, a1, e1) * val

        f32_c_loc_occ = f32_c_loc_occ + cpref * get2(sc_b_ab, sc_b_ab_nc, a0, e0) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, e1, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * val
        f32_x_loc_occ = f32_x_loc_occ + xpref * get2(sc_b_ab, sc_b_ab_nc, a0, e0) * &
            get2(ac_ip_b_left, ac_ip_b_left_nc, e1, 0) * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a1) * val

        f23_c_loc_virt = f23_c_loc_virt + cpref * get2(cs_b_ba, cs_b_ba_nc, e0, a0) * &
            get2(ac_ip_b_right, ac_ip_b_right_nc, e1, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val
        f23_x_ip = f23_x_ip + xpref * get2(aca_b_left, aca_b_left_nc, e0, e1) * &
            get2(sc_ip_b, sc_ip_b_nc, a0, 0) * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a1) * val
      end do
    end subroutine accum_u3_exch_with_ov_b

    subroutine accum_u3_exch(which_frag_is_a, a0, a1, b0, b1, value)
      logical, intent(in) :: which_frag_is_a
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: value
      integer :: e0, e1, idx
      double precision :: term21, term12, term43, term34
      double precision :: t21x, t43x, t43c, t12x, t34x, t34c

      t21x = 0.0d0; t43x = 0.0d0; t43c = 0.0d0
      t12x = 0.0d0; t34x = 0.0d0; t34c = 0.0d0

      if (which_frag_is_a) then
        term21 = value * get2(aca_b_left, aca_b_left_nc, b0, b1)
        term12 = value * get2(aca_b_right, aca_b_right_nc, b0, b1)
        term43 = value * get2(ca_ea_b_left, ca_ea_b_left_nc, 0, b0) * get2(ac_ip_b_right, ac_ip_b_right_nc, b0, 0)
        term34 = value * get2(ca_ea_b_right, ca_ea_b_right_nc, 0, b0) * get2(ac_ip_b_left, ac_ip_b_left_nc, b0, 0)
        do idx = 1, 2
          if (idx == 1) then
            e0 = a1; e1 = a0
          else
            e0 = a0; e1 = a1
          end if
          t21x = t21x + xpref * get2(aca_a_right, aca_a_right_nc, e0, e1) * term21
          t43x = t43x + xpref * get2(ac_ip_a_left, ac_ip_a_left_nc, e0, 0) * &
              get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e1) * term43
          t43c = t43c + cpref * get2(ac_ip_a_left, ac_ip_a_left_nc, e0, 0) * &
              get2(ca_ea_a_right, ca_ea_a_right_nc, 0, e1) * term43
          t12x = t12x + xpref * get2(aca_a_left, aca_a_left_nc, e0, e1) * term12
          t34x = t34x + xpref * get2(ac_ip_a_right, ac_ip_a_right_nc, e0, 0) * &
              get2(ca_ea_a_left, ca_ea_a_left_nc, 0, e1) * term34
          t34c = t34c + cpref * get2(ac_ip_a_right, ac_ip_a_right_nc, e0, 0) * &
              get2(ca_ea_a_left, ca_ea_a_left_nc, 0, e1) * term34
        end do
      else
        term21 = value * get2(aca_a_right, aca_a_right_nc, a0, a1)
        term12 = value * get2(aca_a_left, aca_a_left_nc, a0, a1)
        term43 = value * get2(ca_ea_a_right, ca_ea_a_right_nc, 0, a0) * get2(ac_ip_a_left, ac_ip_a_left_nc, a0, 0)
        term34 = value * get2(ca_ea_a_left, ca_ea_a_left_nc, 0, a0) * get2(ac_ip_a_right, ac_ip_a_right_nc, a0, 0)
        do idx = 1, 2
          if (idx == 1) then
            e0 = b1; e1 = b0
          else
            e0 = b0; e1 = b1
          end if
          t21x = t21x + xpref * get2(aca_b_left, aca_b_left_nc, e0, e1) * term21
          t43x = t43x + xpref * get2(ac_ip_b_right, ac_ip_b_right_nc, e0, 0) * &
              get2(ca_ea_b_left, ca_ea_b_left_nc, 0, e1) * term43
          t43c = t43c + cpref * get2(ac_ip_b_right, ac_ip_b_right_nc, e0, 0) * &
              get2(ca_ea_b_left, ca_ea_b_left_nc, 0, e1) * term43
          t12x = t12x + xpref * get2(aca_b_right, aca_b_right_nc, e0, e1) * term12
          t34x = t34x + xpref * get2(ac_ip_b_left, ac_ip_b_left_nc, e0, 0) * &
              get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e1) * term34
          t34c = t34c + cpref * get2(ac_ip_b_left, ac_ip_b_left_nc, e0, 0) * &
              get2(ca_ea_b_right, ca_ea_b_right_nc, 0, e1) * term34
        end do
      end if

      f43_c = f43_c + t43c
      f21_x = f21_x + t21x
      f43_x = f43_x + t43x
      f12_x = f12_x + t12x
      f34_x = f34_x + t34x
      f34_c = f34_c + t34c
    end subroutine accum_u3_exch

    double precision function new_base(slot, a0, a1, b0, b1, val) result(base)
      integer, intent(in) :: slot, a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: q

      base = 0.0d0
      select case (slot)
      case (57)
        base = get2(scs_a_bb_iq, scs_a_bb_iq_nc, b0, b1) * &
            get2(scs_b_aa_pa, scs_b_aa_pa_nc, a0, a1) * val
      case (69)
        base = get2(cs_a_ab_iq, cs_a_ab_iq_nc, a0, b0) * &
            get2(sc_b_ab_pa, sc_b_ab_pa_nc, a1, b1) * val
      case (75)
        base = get2(sc_a_ba, sc_a_ba_nc, b0, a0) * &
            get2(sc_b_ab_pa, sc_b_ab_pa_nc, a1, b1) * val
      case default
        do q = 0, sc_ip_a_p_nc - 1
          select case (slot)
          case (37)
            base = base + get2(sc_a_ba_pa, sc_a_ba_pa_nc, b0, a0) * &
                get2(ac_ip_a_left, ac_ip_a_left_nc, a1, q) * &
                get2(ca_ea_b_left, ca_ea_b_left_nc, q, b1) * val
          case (39)
            base = base + get2(cs_a_ab_iq, cs_a_ab_iq_nc, a0, b0) * &
                get2(ac_ip_a_left, ac_ip_a_left_nc, a1, q) * &
                get2(ca_ea_b_left, ca_ea_b_left_nc, q, b1) * val
          case (41)
            base = base + get2(aca_a_right, aca_a_right_nc, a0, a1) * &
                get2(sc_ip_a_p, sc_ip_a_p_nc, b0, q) * &
                get2(ca_ea_b_left, ca_ea_b_left_nc, q, b1) * val
          case (43)
            base = base + get2(sc_a_ba_pa, sc_a_ba_pa_nc, b0, a0) * &
                get2(ac_ip_b_right, ac_ip_b_right_nc, b1, q) * &
                get2(ca_ea_a_right, ca_ea_a_right_nc, q, a1) * val
          case (45)
            base = base + get2(aca_a_left, aca_a_left_nc, a0, a1) * &
                get2(ac_ip_b_right, ac_ip_b_right_nc, b0, q) * &
                get2(cs_ea_a_q, cs_ea_a_q_nc, q, b1) * val
          case (47)
            base = base + get2(cs_a_ab_iq, cs_a_ab_iq_nc, a0, b0) * &
                get2(ac_ip_b_right, ac_ip_b_right_nc, b1, q) * &
                get2(ca_ea_a_right, ca_ea_a_right_nc, q, a1) * val
          case (49)
            base = base + get2(scs_a_bb_iq, scs_a_bb_iq_nc, b0, b1) * &
                get2(ca_ea_a_left, ca_ea_a_left_nc, q, a0) * &
                get2(sc_ip_b_p, sc_ip_b_p_nc, a1, q) * val
          case (51)
            base = base + get2(scs_a_bb_pa, scs_a_bb_pa_nc, b0, b1) * &
                get2(ca_ea_a_right, ca_ea_a_right_nc, q, a0) * &
                get2(sc_ip_b, sc_ip_b_nc, a1, q) * val
          case (53)
            base = base + get2(scs_a_bb_iq, scs_a_bb_iq_nc, b0, b1) * &
                get2(cs_ea_b, cs_ea_b_nc, q, a0) * &
                get2(ac_ip_a_left, ac_ip_a_left_nc, a1, q) * val
          case (55)
            base = base + get2(scs_a_bb_pa, scs_a_bb_pa_nc, b0, b1) * &
                get2(cs_ea_b_q, cs_ea_b_q_nc, q, a0) * &
                get2(ac_ip_a_right, ac_ip_a_right_nc, a1, q) * val
          case (59)
            base = base + get2(ca_ea_b_left, ca_ea_b_left_nc, q, b0) * &
                get2(sc_ip_b, sc_ip_b_nc, a0, q) * &
                get2(ca_ea_a_right, ca_ea_a_right_nc, q, a1) * &
                get2(sc_ip_a_p, sc_ip_a_p_nc, b1, q) * val
          case (61)
            base = base + get2(cs_a_ab_iq, cs_a_ab_iq_nc, a0, b0) * &
                get2(cs_ea_a, cs_ea_a_nc, q, b1) * &
                get2(sc_ip_b_p, sc_ip_b_p_nc, a1, q) * val
          case (63)
            base = base + get2(cs_a_ab, cs_a_ab_nc, a0, b0) * &
                get2(cs_ea_a_q, cs_ea_a_q_nc, q, b1) * &
                get2(sc_ip_b, sc_ip_b_nc, a1, q) * val
          case (65)
            base = base + get2(sc_a_ba, sc_a_ba_nc, b0, a0) * &
                get2(cs_ea_b, cs_ea_b_nc, q, a1) * &
                get2(sc_ip_a_p, sc_ip_a_p_nc, b1, q) * val
          case (67)
            base = base + get2(sc_a_ba_pa, sc_a_ba_pa_nc, b0, a0) * &
                get2(cs_ea_b_q, cs_ea_b_q_nc, q, a1) * &
                get2(sc_ip_a, sc_ip_a_nc, b1, q) * val
          case (71)
            base = base + get2(sc_a_ba, sc_a_ba_nc, b0, a0) * &
                get2(cs_ea_a, cs_ea_a_nc, q, b1) * &
                get2(sc_ip_b_p, sc_ip_b_p_nc, a1, q) * val
          case (73)
            base = base + get2(cs_a_ab, cs_a_ab_nc, a0, b0) * &
                get2(cs_ea_b_q, cs_ea_b_q_nc, q, a1) * &
                get2(sc_ip_a, sc_ip_a_nc, b1, q) * val
          case (77)
            base = base + get2(cs_ea_b, cs_ea_b_nc, q, a0) * &
                get2(ac_ip_b_right, ac_ip_b_right_nc, b0, q) * &
                get2(ca_ea_a_right, ca_ea_a_right_nc, q, a1) * &
                get2(sc_ip_a_p, sc_ip_a_p_nc, b1, q) * val
          case (79)
            base = base + get2(cs_ea_b, cs_ea_b_nc, q, a0) * &
                get2(ac_ip_b_right, ac_ip_b_right_nc, b0, q) * &
                get2(cs_ea_a_q, cs_ea_a_q_nc, q, b1) * &
                get2(ac_ip_a_left, ac_ip_a_left_nc, a1, q) * val
          case (81)
            base = base + get2(ac_ip_a_left, ac_ip_a_left_nc, a0, q) * &
                get2(ca_ea_b_left, ca_ea_b_left_nc, q, b0) * &
                get2(sc_ip_b, sc_ip_b_nc, a1, q) * &
                get2(cs_ea_a_q, cs_ea_a_q_nc, q, b1) * val
          case (137)
            base = base + get2(sc_a_ba_pa, sc_a_ba_pa_nc, b0, a0) * &
                get2(ac_ip_a_right, ac_ip_a_right_nc, a1, q) * &
                get2(ca_ea_b_right, ca_ea_b_right_nc, q, b1) * val
          case (147)
            base = base + get2(cs_a_ab_iq, cs_a_ab_iq_nc, a0, b0) * &
                get2(ac_ip_b_left, ac_ip_b_left_nc, b1, q) * &
                get2(ca_ea_a_left, ca_ea_a_left_nc, q, a1) * val
          end select
        end do
      end select
    end function new_base

    subroutine add_new(slot, is_coul, a0, a1, b0, b1, val)
      integer, intent(in) :: slot, a0, a1, b0, b1
      logical, intent(in) :: is_coul
      double precision, intent(in) :: val
      if (is_coul) then
        out(slot) = out(slot) + cpref * new_base(slot, a0, a1, b0, b1, val)
      else
        out(slot + 1) = out(slot + 1) + xpref * new_base(slot, a0, a1, b0, b1, val)
      end if
    end subroutine add_new

    subroutine add_new_variant(output_slot, formula_slot, is_coul, a0, a1, b0, b1, val)
      integer, intent(in) :: output_slot, formula_slot, a0, a1, b0, b1
      logical, intent(in) :: is_coul
      double precision, intent(in) :: val
      if (is_coul) then
        out(output_slot) = out(output_slot) + cpref * new_base(formula_slot, a0, a1, b0, b1, val)
      else
        out(output_slot + 1) = out(output_slot + 1) + xpref * new_base(formula_slot, a0, a1, b0, b1, val)
      end if
    end subroutine add_new_variant

    subroutine accum_new_u2_coul(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      call add_new(81, .false., a0, a1, b0, b1, val)
      call add_new(49, .true.,  a0, a1, b0, b1, val)
      call add_new(51, .true.,  a0, a1, b0, b1, val)
      call add_new(53, .true.,  a0, a1, b0, b1, val)
      call add_new(55, .true.,  a0, a1, b0, b1, val)
      call add_new(57, .true.,  a0, a1, b0, b1, val)
      call add_new(59, .true.,  a0, a1, b0, b1, val)
      call add_new(61, .false., a0, a1, b0, b1, val)
      call add_new(63, .false., a0, a1, b0, b1, val)
      call add_new(65, .false., a0, a1, b0, b1, val)
      call add_new(67, .false., a0, a1, b0, b1, val)
      call add_new(69, .false., a0, a1, b0, b1, val)
      call add_new(77, .false., a0, a1, b0, b1, val)
      call add_new(79, .true.,  a0, a1, b0, b1, val)
      call add_new(41, .true.,  a0, a1, b0, b1, val)
      call add_new(39, .false., a0, a1, b0, b1, val)
      call add_new(45, .true.,  a0, a1, b0, b1, val)
      call add_new(43, .false., a1, a0, b0, b1, val)
    end subroutine accum_new_u2_coul

    subroutine accum_new_u2_exch(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      call add_new(81, .true.,  a0, a1, b0, b1, val)
      call add_new(49, .false., a0, a1, b0, b1, val)
      call add_new(51, .false., a0, a1, b0, b1, val)
      call add_new(53, .false., a0, a1, b0, b1, val)
      call add_new(55, .false., a0, a1, b0, b1, val)
      call add_new(57, .false., a0, a1, b0, b1, val)
      call add_new(59, .false., a0, a1, b0, b1, val)
      call add_new(61, .true.,  a0, a1, b0, b1, val)
      call add_new(63, .true.,  a0, a1, b0, b1, val)
      call add_new(65, .true.,  a0, a1, b0, b1, val)
      call add_new(67, .true.,  a0, a1, b0, b1, val)
      call add_new(69, .true.,  a0, a1, b0, b1, val)
      call add_new(71, .true.,  a0, a1, b0, b1, val)
      call add_new(71, .false., a0, a1, b0, b1, val)
      call add_new(73, .true.,  a0, a1, b0, b1, val)
      call add_new(73, .false., a0, a1, b0, b1, val)
      call add_new(75, .true.,  a0, a1, b0, b1, val)
      call add_new(75, .false., a0, a1, b0, b1, val)
      call add_new(77, .true.,  a0, a1, b0, b1, val)
      call add_new(79, .false., a0, a1, b0, b1, val)
      call add_new(39, .true.,  a0, a1, b0, b1, val)
      call add_new(41, .false., a0, a1, b0, b1, val)
      call add_new(43, .true.,  a1, a0, b0, b1, val)
      call add_new(45, .false., a0, a1, b0, b1, val)
      call add_new(47, .true.,  a0, a1, b0, b1, val)
      call add_new(47, .false., a0, a1, b0, b1, val)
      call add_new(37, .true.,  a0, a1, b0, b1, val)
      call add_new(37, .false., a0, a1, b0, b1, val)
    end subroutine accum_new_u2_exch

    subroutine accum_new_u3_coul(repeat_a, a0, a1, b0, b1, val)
      logical, intent(in) :: repeat_a
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: e0, e1, pa0, pa1, pb0, pb1, idx
      do idx = 1, 2
        if (repeat_a) then
          if (idx == 1) then
            e0 = a1; e1 = a0
          else
            e0 = a0; e1 = a1
          end if
          pa0 = e0; pa1 = e1; pb0 = b0; pb1 = b1
        else
          if (idx == 1) then
            e0 = b1; e1 = b0
          else
            e0 = b0; e1 = b1
          end if
          pa0 = a0; pa1 = a1; pb0 = e0; pb1 = e1
        end if
        call add_new(81, .false., pa0, pa1, pb0, pb1, val)
        call add_new(49, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(51, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(53, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(55, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(57, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(59, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(61, .false., pa0, pa1, pb0, pb1, val)
        call add_new(63, .false., pa0, pa1, pb0, pb1, val)
        call add_new(65, .false., pa0, pa1, pb0, pb1, val)
        call add_new(67, .false., pa0, pa1, pb0, pb1, val)
        call add_new(69, .false., pa0, pa1, pb0, pb1, val)
        call add_new(77, .false., pa0, pa1, pb1, pb0, val)
        if (repeat_a) then
          call add_new(79, .true., pa1, pa0, pb0, pb1, val)
        else
          call add_new(79, .true., pa0, pa1, pb0, pb1, val)
        end if
        call add_new(41, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(39, .false., pa0, pa1, pb0, pb1, val)
        call add_new(45, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(43, .false., pa1, pa0, pb0, pb1, val)
      end do
    end subroutine accum_new_u3_coul

    subroutine accum_new_u3_exch(repeat_a, a0, a1, b0, b1, val)
      logical, intent(in) :: repeat_a
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: e0, e1, pa0, pa1, pb0, pb1, xa0, xa1, xb0, xb1, idx
      do idx = 1, 2
        if (repeat_a) then
          if (idx == 1) then
            e0 = a1; e1 = a0
          else
            e0 = a0; e1 = a1
          end if
          pa0 = e0; pa1 = e1; pb0 = b0; pb1 = b1
          xa0 = pa1; xa1 = pa0; xb0 = pb0; xb1 = pb1
        else
          if (idx == 1) then
            e0 = b1; e1 = b0
          else
            e0 = b0; e1 = b1
          end if
          pa0 = a0; pa1 = a1; pb0 = e0; pb1 = e1
          xa0 = pa0; xa1 = pa1; xb0 = pb1; xb1 = pb0
        end if
        call add_new(81, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(49, .false., pa1, pa0, pb0, pb1, val)
        call add_new(51, .false., pa1, pa0, pb0, pb1, val)
        call add_new(53, .false., pa1, pa0, pb0, pb1, val)
        call add_new(55, .false., pa1, pa0, pb0, pb1, val)
        call add_new(57, .false., pa0, pa1, pb0, pb1, val)
        call add_new(59, .false., pa0, pa1, pb0, pb1, val)
        call add_new(61, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(63, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(65, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(67, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(69, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(71, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(71, .false., xa0, xa1, xb0, xb1, val)
        call add_new(73, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(73, .false., xa0, xa1, xb0, xb1, val)
        call add_new(75, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(75, .false., xa0, xa1, xb0, xb1, val)
        call add_new(77, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(79, .false., pa0, pa1, pb0, pb1, val)
        call add_new(39, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(41, .false., pa0, pa1, pb0, pb1, val)
        call add_new(43, .true.,  pa1, pa0, pb0, pb1, val)
        call add_new(45, .false., pa0, pa1, pb0, pb1, val)
        call add_new(47, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(47, .false., xa0, xa1, xb0, xb1, val)
        call add_new(37, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(37, .false., xa0, xa1, xb0, xb1, val)
      end do
    end subroutine accum_new_u3_exch

    subroutine accum_new_u4_coul(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: pa0, pa1, pb0, pb1, ia, ib
      do ia = 1, 2
        if (ia == 1) then
          pa0 = a1; pa1 = a0
        else
          pa0 = a0; pa1 = a1
        end if
        do ib = 1, 2
          if (ib == 1) then
            pb0 = b1; pb1 = b0
          else
            pb0 = b0; pb1 = b1
          end if
          call add_new(81, .false., pa0, pa1, pb0, pb1, val)
          call add_new(77, .false., pa1, pa0, pb1, pb0, val)
          call add_new(79, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(49, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(51, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(53, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(55, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(57, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(59, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(61, .false., pa0, pa1, pb0, pb1, val)
          call add_new(63, .false., pa0, pa1, pb0, pb1, val)
          call add_new(65, .false., pa0, pa1, pb0, pb1, val)
          call add_new(67, .false., pa0, pa1, pb0, pb1, val)
          call add_new(69, .false., pa0, pa1, pb0, pb1, val)
          call add_new(41, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(39, .false., pa0, pa1, pb0, pb1, val)
          call add_new(45, .true.,  pa0, pa1, pb0, pb1, val)
          call add_new(43, .false., pa0, pa1, pb0, pb1, val)
        end do
      end do
    end subroutine accum_new_u4_coul

    subroutine accum_new_u4_exch(a0, a1, b0, b1, val)
      integer, intent(in) :: a0, a1, b0, b1
      double precision, intent(in) :: val
      integer :: pa0, pa1, pb0, pb1, idx
      do idx = 1, 2
        if (idx == 1) then
          pa0 = a1; pa1 = a0; pb0 = b1; pb1 = b0
        else
          pa0 = a0; pa1 = a1; pb0 = b0; pb1 = b1
        end if
        call add_new(81, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(49, .false., pa1, pa0, pb0, pb1, val)
        call add_new(51, .false., pa1, pa0, pb0, pb1, val)
        call add_new(53, .false., pa1, pa0, pb0, pb1, val)
        call add_new(55, .false., pa1, pa0, pb0, pb1, val)
        call add_new(57, .false., pa0, pa1, pb0, pb1, val)
        call add_new(59, .false., pa0, pa1, pb1, pb0, val)
        call add_new(61, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(63, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(65, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(67, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(69, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(71, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(71, .false., pa1, pa0, pb0, pb1, val)
        call add_new(73, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(73, .false., pa0, pa1, pb1, pb0, val)
        call add_new(75, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(75, .false., pa0, pa1, pb1, pb0, val)
        call add_new(77, .true.,  pa0, pa1, pb1, pb0, val)
        call add_new(79, .false., pa1, pa0, pb0, pb1, val)
        call add_new(39, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(41, .false., pa0, pa1, pb0, pb1, val)
        call add_new(43, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new(45, .false., pa0, pa1, pb0, pb1, val)
        call add_new_variant(47, 147, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new_variant(47, 147, .false., pa0, pa1, pb1, pb0, val)
        call add_new_variant(37, 137, .true.,  pa0, pa1, pb0, pb1, val)
        call add_new_variant(37, 137, .false., pa1, pa0, pb0, pb1, val)
      end do
    end subroutine accum_new_u4_exch

  end subroutine bbaa_accum_all
end module twoelint_bbaa_mod
